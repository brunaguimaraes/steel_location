#==============================================================================
# 1. IMPORTS AND PATHS
#==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import (
    ConcreteModel, Set, Var, Param, Constraint, Objective,
    NonNegativeReals, Binary, minimize, value,
)
from pyomo.environ import NonNegativeIntegers
from pyomo.opt import SolverFactory
from amplpy import modules
modules.install("highs")

# >>> CHANGE ONLY THIS LINE WHEN MOVING TO ANOTHER MACHINE <<<
BASE_DIR =  r"C:\Users\ottoh\OneDrive\ARTIGO\submission\Research Data"
# BASE_DIR = r"C:/Users/Bruna/OneDrive/DOUTORADO/0.TESE/modelagem/modelo_bru"

PLANTS_FILE  = os.path.join(BASE_DIR, "input/Input_existing_plants.xlsx")
CONFIG_FILE  = os.path.join(BASE_DIR, "input/Input_Model_Config_12_horizonte_2070.xlsx")
OUTPUT_DIR   = os.path.join(BASE_DIR, "resultados")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# LOC ========================================================================
# 1a. STATE / REGION DEFINITIONS AND AVAILABILITY RULES  (LOC layer)
# ============================================================================
# Brazilian regions used to restrict WHERE certain routes may be built. Only
# affects NEW capacity (successor slots and greenfield) — existing plants keep
# operating wherever they physically are.
SUDESTE_STATES  = {"SP", "RJ", "MG", "ES"}
NORDESTE_STATES = {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}

# All 27 Brazilian UFs — default greenfield candidate set.
ALL_BR_STATES = {
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO", "MA", "MT", "MS",
    "MG", "PA", "PB", "PR", "PE", "PI", "RJ", "RN", "RS", "RO", "RR", "SC",
    "SP", "SE", "TO",
}

# CCS (BF-BOF-CCS): the full Sudeste region (storage-geology premise).
CCS_ALLOWED_STATES = set(SUDESTE_STATES)
# Natural gas (DR-NG): Southeast + Northeast.
GN_ALLOWED_STATES = SUDESTE_STATES | NORDESTE_STATES
# Green H2 (DR-H2): announced low-carbon H2 hubs (frozen 07/2026).
H2_ALLOWED_STATES = {"CE", "PE", "RN", "PI", "MA",   # Nordeste
                     "RJ", "ES",                     # Sudeste
                     "RS"}                            # Sul

# DR-NG-CCS (LOC decision): needs BOTH gas AND CCS storage, so its allowed
# set is the intersection = {ES, MG, RJ, SP}. Registered as a methodological
# premise to revisit with the advisor (the CCS=Sudeste storage assumption is
# the one most likely to be widened, e.g. to onshore basins such as BA).
DRNGCCS_ALLOWED_STATES = GN_ALLOWED_STATES & CCS_ALLOWED_STATES

# Route -> allowed-state set for NEW capacity (names post-sanitization).
ROUTE_STATE_RESTRICTIONS = {
    "BF-BOF-CCS": CCS_ALLOWED_STATES,
    "DR-NG":      GN_ALLOWED_STATES,
    "DR-H2":      H2_ALLOWED_STATES,
    "DR-NG-CCS":  DRNGCCS_ALLOWED_STATES,   # LOC: gas ∩ CCS = {ES, MG, RJ, SP}
    # Unlisted routes (BF-BOF_MC, BF-BOF_CC, EAF, IBT) have no restriction.
}

CHARCOAL_SUPPLY_STATE_SHEET = "Charcoal_Supply_State"  # reserved for later use
# LOC end ====================================================================


#==============================================================================
# 1b. CHARCOAL SUPPLY CURVE (3 cumulative tiers)
#==============================================================================
# The charcoal supply curve is a step function with three cumulative tiers:
#
#   tier 1 — first  50 PJ/year   at  9.80 USD/GJ
#   tier 2 — next   25 PJ/year   at 15.00 USD/GJ   (cumulative range 50..75)
#   tier 3 — next   25 PJ/year   at 20.00 USD/GJ   (cumulative range 75..100)
#   above 100 PJ/year: not available (hard supply cap)
#
# Demand of D PJ pays:  tier 1 width * tier 1 price + ... up to D
# This is a marginal-cost (convex, increasing) supply curve, so the segment-
# increment linearization needs NO binary variables — the solver fills the
# cheap tier first by itself when minimizing cost.
#
# This DEPRECATES the previous endogenous/combined price scheme and IGNORES
# the charcoal column of the Fuel_Prices sheet — the supply curve is now
# defined entirely here.
#
# Tiers are expressed in PJ/year (1 PJ = 1e6 GJ). Edit the lists below to
# change widths or prices. Add or remove rows to change the number of tiers.

CHARCOAL_FUEL = "Carvao_vegetal"

# Tier widths in PJ/year and tier prices in USD/GJ.
CHARCOAL_TIER_WIDTH_PJ = [25.0, 10.0, 10.0, 10.0, 10.0, 10.0, 25.0, 50.0]
CHARCOAL_TIER_PRICE    = [9.8, 13.0, 16.0, 19.0, 22.0, 25.0, 28.0, 32.0]




# What happens above the highest tier (total width = sum of widths above)?
#   "hard_cap" -> demand is forbidden to exceed the curve; model becomes
#                 infeasible if it tries. Useful to model an exhausted supply.
#   "extend"   -> add an extra tier at the same price as the last one,
#                 sized as EXTEND_WIDTH_PJ, so the model can still pay
#                 (at the top price) above the cap.
CHARCOAL_ABOVE_CAP = "extend"
CHARCOAL_EXTEND_WIDTH_PJ = 1000.0   # only used if CHARCOAL_ABOVE_CAP == "extend"


#==============================================================================
# 1c. TECHNOLOGY PENETRATION LIMITS (CCS and H2 ramp-in)
#==============================================================================
# Restrict how fast certain technologies can enter the steel mix. For each
# constrained route, the total annual production (across successor slots and
# greenfield) is capped at a fraction of that year's production target:
#
#     production_route[y]  <=  s(y) * production_target[y]
#
# where s(y) is a smoothstep sigmoid that equals 0 at PENETRATION_YEAR_START
# and 1 at PENETRATION_YEAR_END, with exact anchoring at both ends:
#
#     y <= START        -> s = 0
#     START < y < END   -> t = (y-START)/(END-START);  s = 3t^2 - 2t^3
#     y >= END          -> s = 1
#
# Each entry in PENETRATION_LIMITS gives a separate sigmoid for one route.
# To make the limit a JOINT cap (e.g. CCS+H2 share <= s(y)), use a list of
# routes as the key — see the commented example below.

PENETRATION_LIMITS = {
    "BF-BOF-CCS": {"start": 2035, "end": 2050},
    "DR-NG-CCS": {"start": 2035, "end": 2050},
    "DR-H2":      {"start": 2035, "end": 2050},
    # Example: joint cap on CCS+H2 combined would be:
    # ("BF-BOF-CCS", "DR-H2"): {"start": 2035, "end": 2050},
}

#==============================================================================
# 1d. FUEL CATEGORIES
#==============================================================================

# Fuel categories for the energy-mix indicator.
FUEL_CATEGORY = {
    "Eletricidade":    "Electricity",
    "Carvao_mineral":  "Fossil",
    "Coque":           "Fossil",
    "Gas_de_coqueria": "Fossil",
    "Gas_natural":     "Fossil",
    "Oleo_diesel":     "Fossil",
    "Carvao_vegetal":  "Renewable",
    "Hidrogenio":      "Renewable",
}

# ===================================================================
# CCS ROUTES
# ===================================================================
# Every technology route that captures carbon should be listed here.
# CAPTURE_RATE_CCS (a single scalar from Model_Config.xlsx, Parameters
# sheet) is applied to ALL routes in this set. If different CCS routes
# need different capture rates in the future, this should become a
# per-route dict instead (e.g. read from a new column in the Routes
# sheet) — flag it if that's needed.
CCS_ROUTES = {"BF-BOF-CCS", "DR-NG-CCS"}

# ===================================================================
# COLOR PALETTE — consistent across MIT and REF scenarios
# Mirrors the matplotlib tab10 default order used in the MIT plots
# ===================================================================
ROUTE_COLORS = {
    "BF-BOF-CCS":  "#1f77b4",   # tab:blue
    "BF-BOF_CC":   "#ff7f0e",   # tab:orange
    "BF-BOF_MC":   "#2ca02c",   # tab:green
    "DR-H2":       "#d62728",   # tab:red
    "DR-NG":       "#9467bd",   # tab:purple
    "EAF":         "#8c564b",   # tab:brown
    "IBT":         "#e377c2",   # tab:pink
    "DR-NG-CCS":  "#7f7f7f",   # tab:gray (NEW)
}

FUEL_COLORS = {
    "Coal":         "#1f77b4",   # tab:blue
    "Charcoal":     "#ff7f0e",   # tab:orange
    "Electricity":  "#2ca02c",   # tab:green
    "Natural Gas":  "#d62728",   # tab:red
    "Diesel":       "#9467bd",   # tab:purple (preserva o MIT)
    "Hydrogen":     "#17becf",   # tab:cyan (cor distinta)
    "Coke":         "#8c564b",   # tab:brown
    "Coke Oven Gas":"#e377c2",   # tab:pink
    "Scrap":        "#bcbd22",   # tab:olive
}
#==============================================================================
# 2. LOAD CONFIG
#==============================================================================

def _sanitize_spaces(cfg: dict) -> dict:
    """Replace spaces with underscores in all route/fuel names to avoid NL format issues."""
    def fix(s):
        return s.replace(" ", "_") if isinstance(s, str) else s

    cfg["routes"] = [fix(r) for r in cfg["routes"]]
    cfg["capex_retrofit"]   = {fix(k): v for k, v in cfg["capex_retrofit"].items()}
    cfg["capex_greenfield"] = {fix(k): v for k, v in cfg["capex_greenfield"].items()}
    cfg["opex_fixed"] = {fix(k): v for k, v in cfg["opex_fixed"].items()}
    cfg["scrap_rate"] = {fix(k): v for k, v in cfg["scrap_rate"].items()}
    cfg["uses_biomass"] = {fix(k): v for k, v in cfg["uses_biomass"].items()}
    cfg["greenfield_max_capacity"] = {fix(k): v for k, v in cfg["greenfield_max_capacity"].items()}
    cfg["ei"] = {(fix(r), fix(f)): v for (r, f), v in cfg["ei"].items()}
    cfg["fuels_by_route"] = {fix(r): [fix(f) for f in fs] for r, fs in cfg["fuels_by_route"].items()}
    cfg["prices"] = {(fix(f), y): v for (f, y), v in cfg["prices"].items()}
    cfg["prices_state"] = {(fix(f), s): v for (f, s), v in cfg.get("prices_state", {}).items()}  # LOC
    cfg["ore_delta"]    = {(fix(r), s): v for (r, s), v in cfg.get("ore_delta", {}).items()}      # LOC
    cfg["ef"] = {fix(k): v for k, v in cfg["ef"].items()}
    return cfg


def load_config(path: str) -> dict:
    """Load all sheets from Model_Config.xlsx into one dict."""
    cfg = {}

    # ---- Scalar parameters
    df_p = pd.read_excel(path, sheet_name="Parameters")
    params = dict(zip(df_p["Parameter"], df_p["Value"]))
    cfg["YEAR_START"]          = int(params["YEAR_START"])
    cfg["YEAR_END"]            = int(params["YEAR_END"])
    cfg["BASE_YEAR"]           = int(params["BASE_YEAR"])
    cfg["DISCOUNT_RATE"]       = float(params["DISCOUNT_RATE"])
    cfg["MIN_UTILIZATION"]     = float(params["MINIMUM_UTILIZATION"])
    cfg["PLANT_LIFETIME"]      = int(params["PLANT_LIFETIME"])
    cfg["CAPEX_AMORTIZATION"]  = int(params["CAPEX_AMORTIZATION"])
    cfg["CAPACITY_MULTIPLIER"] = float(params["CAPACITY_MULTIPLIER"])
    cfg["CAPTURE_RATE_CCS"]    = float(params["CAPTURE_RATE_CCS"])
    # Maximum allowed year-on-year production drop (0.10 = 10%)
    cfg["MAX_RAMP_DOWN"]       = float(params.get("MAX_RAMP_DOWN", 0.10))
    cfg["YEARS"] = list(range(cfg["YEAR_START"], cfg["YEAR_END"] + 1))

    # ---- Routes
    df_r = pd.read_excel(path, sheet_name="Routes")
    cfg["routes"] = df_r["Route"].tolist()
    # CAPEX differentiated by pathway: retrofit (successor slots, cheaper —
    # reuses site/foundation/utilities) vs greenfield (new plant, pricier).
    cfg["capex_retrofit"]   = dict(zip(df_r["Route"], df_r["CAPEX_retrofit_USD_per_t"]))
    cfg["capex_greenfield"] = dict(zip(df_r["Route"], df_r["CAPEX_greenfield_USD_per_t"]))
    cfg["opex_fixed"] = dict(zip(df_r["Route"], df_r["OPEX_fixed_USD_per_t"]))
    cfg["scrap_rate"] = dict(zip(df_r["Route"], df_r["Scrap_rate_t_per_t"]))
    cfg["uses_biomass"] = {
        r: str(v).strip().upper() == "TRUE"
        for r, v in zip(df_r["Route"], df_r["Uses_biomass"])
    }
    cfg["greenfield_max_capacity"] = dict(
        zip(df_r["Route"], df_r["Greenfield_max_capacity_kt"].astype(float))
    )

    # ---- Route × Fuel consumption (GJ/t)
    df_ei = pd.read_excel(path, sheet_name="Route_Fuel_Consumption")
    cfg["ei"] = {
        (row["Route"], row["Fuel"]): float(row["EI_GJ_per_t"])
        for _, row in df_ei.iterrows()
    }
    cfg["fuels_by_route"] = {
        r: df_ei.loc[df_ei["Route"] == r, "Fuel"].tolist()
        for r in cfg["routes"]
    }

    # ---- Fuel prices
    df_pr = pd.read_excel(path, sheet_name="Fuel_Prices")
    df_pr = df_pr.set_index("Fuel")
    df_pr.columns = [int(c) for c in df_pr.columns]
    cfg["prices"] = {
        (fuel, year): float(df_pr.loc[fuel, year])
        for fuel in df_pr.index
        for year in df_pr.columns
    }

    # LOC ---- Fuel prices BY STATE (fixed, no year dimension).
    # Sheet "Fuel_Prices_State": columns Fuel, State, Price_USD_per_GJ.
    try:
        df_pr_state = pd.read_excel(path, sheet_name="Fuel_Prices_State")
        df_pr_state.columns = [c.strip() for c in df_pr_state.columns]
        cfg["prices_state"] = {
            (str(row["Fuel"]).strip(), str(row["State"]).strip().upper()):
                float(row["Price_USD_per_GJ"])
            for _, row in df_pr_state.iterrows()
        }
    except ValueError:
        print("    [warn] Sheet 'Fuel_Prices_State' not found — electricity/"
              "natural gas will use the national fixed price for every state.")
        cfg["prices_state"] = {}

    # LOC ---- Ore/pellet state differential (DELTAS vs. route anchor).
    # Sheet "Ore_Cost_State": columns Route, State, Extra_cost_USD_per_t.
    try:
        df_ore = pd.read_excel(path, sheet_name="Ore_Cost_State")
        df_ore.columns = [c.strip() for c in df_ore.columns]
        cfg["ore_delta"] = {
            (str(row["Route"]).strip(), str(row["State"]).strip().upper()):
                float(row["Extra_cost_USD_per_t"])
            for _, row in df_ore.iterrows()
            if pd.notna(row["Extra_cost_USD_per_t"])
        }
    except ValueError:
        print("    [warn] Sheet 'Ore_Cost_State' not found — ore/pellet cost "
              "carries no state differential.")
        cfg["ore_delta"] = {}

    # LOC ---- Candidate states for GREENFIELD (optional sheet).
    # Sheet "Greenfield_States": single column "UF". If absent, build_model
    # falls back to the unique UFs present among existing plants / all UFs.
    try:
        df_gs = pd.read_excel(path, sheet_name="Greenfield_States")
        df_gs.columns = [c.strip() for c in df_gs.columns]
        cfg["greenfield_states"] = sorted({
            str(u).strip().upper() for u in df_gs["UF"] if pd.notna(u)
        })
    except ValueError:
        cfg["greenfield_states"] = None

    # ---- Emission factors
    df_ef = pd.read_excel(path, sheet_name="Emission_Factors")
    cfg["ef"] = dict(zip(df_ef["Fuel"], df_ef["EF_tCO2_per_GJ"].astype(float)))

    # ---- Yearly series
    cfg["emission_cap"]      = _series(path, "Emission_Cap",      "Emission_cap_tCO2")
    cfg["production_target"] = _series(path, "Production_Target", "Production_kt")
    cfg["scrap_supply"]      = _series(path, "Scrap_Supply",      "Scrap_supply_kt")
    cfg["biomass_supply"]    = _series(path, "Biomass_Supply",    "Biomass_supply_GJ")
    # Max fossil-fuel share of the aggregate energy mix (fraction 0-1).
    # NEW: fossil fuel share constraint (see CONSTRAINT I in build_model).
    cfg["fossil_share_max"]  = _series(path, "Fossil_Share_Max",  "Fossil_share_max")

    # LOC ---- Regional scrap: shares + UF->region map.
    # Two sheets regionalize the scrap constraint (Constraint D). See
    # Scrap_Regional_Config.xlsx and the methodology note.
    #   Scrap_Regional_Share: columns Region, Share (fraction; should sum to 1).
    #   UF_Region:            columns UF, Region.
    df_share = pd.read_excel(path, sheet_name="Scrap_Regional_Share")
    df_share.columns = [c.strip() for c in df_share.columns]
    cfg["scrap_share"] = {
        str(row["Region"]).strip(): float(row["Share"])
        for _, row in df_share.iterrows()
        if pd.notna(row.get("Region")) and pd.notna(row.get("Share"))
    }
    _ssum = sum(cfg["scrap_share"].values())
    if abs(_ssum - 1.0) > 1e-6:
        print(f"    [warn] Scrap_Regional_Share shares sum to {_ssum:.4f}, not 1.0.")

    df_ufreg = pd.read_excel(path, sheet_name="UF_Region")
    df_ufreg.columns = [c.strip() for c in df_ufreg.columns]
    cfg["uf_region"] = {
        str(row["UF"]).strip().upper(): str(row["Region"]).strip()
        for _, row in df_ufreg.iterrows()
        if pd.notna(row.get("UF")) and pd.notna(row.get("Region"))
    }
    _missing = set(cfg["uf_region"].values()) - set(cfg["scrap_share"])
    if _missing:
        raise ValueError(
            f"Regions in UF_Region without a Share in Scrap_Regional_Share: "
            f"{sorted(_missing)}"
        )

    cfg = _sanitize_spaces(cfg)

    # ---- Sanity check: warn if any CCS route in CCS_ROUTES is missing from
    #      the loaded Routes sheet (keeps the two config sources in sync).
    missing_ccs = [r for r in CCS_ROUTES if r not in cfg["routes"]]
    if missing_ccs:
        print(f"    [warn] CCS_ROUTES contains route(s) not found in the "
              f"'Routes' sheet: {missing_ccs}. They will be ignored.")

    return cfg


def _series(path: str, sheet: str, value_col: str) -> dict:
    df = pd.read_excel(path, sheet_name=sheet)
    return {int(y): float(v) for y, v in zip(df["Year"], df[value_col])}


#==============================================================================
# 3. LOAD PLANTS
#==============================================================================

def load_plants(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip() for c in df.columns]

    if "Route_detailed" in df.columns and "Route" in df.columns:
        df = df.drop(columns=["Route"])

    rename = {
        "Plantname":      "PlantID",
        "Route_detailed": "Route",
        "Capacity":       "Capacity",
        "Startyear":      "Startyear",
        "Retrofitdate":   "Retrofitdate",
    }
    df = df.rename(columns=rename)

    route_map = {
        "BF-BOF CM": "BF-BOF_MC",
        "BF-BOF CV": "BF-BOF_CC",
    }
    df["Route"] = df["Route"].replace(route_map)
    df["PlantID"] = df["PlantID"].str.replace(" ", "_")

    # LOC ---- UF (state): required for the locational layer. Prefer "UF",
    # accept "State" as fallback, normalize to 2-letter uppercase code.
    if "UF" not in df.columns and "State" in df.columns:
        df = df.rename(columns={"State": "UF"})

    required = ["PlantID", "Route", "Capacity", "Startyear", "Retrofitdate", "UF"]  # LOC: +UF
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Plants file is missing required column(s): {missing}. "
            f"Found columns: {df.columns.tolist()}. "
            f"'UF' must be a 2-letter Brazilian state code per plant."
        )

    df["Startyear"]    = df["Startyear"].astype(int)
    df["Retrofitdate"] = df["Retrofitdate"].astype(int)
    df["Capacity"]     = df["Capacity"].astype(float)
    df["UF"]           = df["UF"].astype(str).str.strip().str.upper()  # LOC

    bad_uf = df[~df["UF"].str.match(r"^[A-Z]{2}$")]  # LOC
    if len(bad_uf) > 0:
        raise ValueError(
            f"Invalid UF value(s) in existing_plants: "
            f"{bad_uf[['PlantID', 'UF']].to_dict('records')}"
        )

    return df[required]


#==============================================================================
# 4. PRECOMPUTE ROUTE-LEVEL FACTORS AND COSTS
#==============================================================================

def compute_route_emission_factor(cfg: dict) -> dict:
    """EF_route[r] = Σ_fuel  EI[r,f] · EF[f]   (tCO2 per t of steel)
    CCS captures CAPTURE_RATE_CCS of the total emissions for every route
    listed in CCS_ROUTES (currently BF-BOF-CCS and DR-NG-CCS).
    """
    ef_route = {}
    for r in cfg["routes"]:
        total = 0.0
        for f in cfg["fuels_by_route"][r]:
            total += cfg["ei"].get((r, f), 0.0) * cfg["ef"].get(f, 0.0)
        if r in CCS_ROUTES:
            total = total * (1 - cfg["CAPTURE_RATE_CCS"])
        ef_route[r] = total
    return ef_route


# LOC: fuels whose price is state-dependent. Any fuel here looks up
# cfg["prices_state"][(fuel, state)] first, falling back to the national
# cfg["prices"][(fuel, year)] price if the state isn't in the sheet.
STATE_PRICED_FUELS = {"Eletricidade", "Gas_natural", "Carvao_mineral",
                      "Oleo_diesel", "Hidrogenio"}


def compute_route_fuel_cost_state(cfg: dict, state: str) -> dict:
    """LOC: fixed-price fuel cost per (route, year), USD/t of steel, for a
    unit located in `state`. Eletricidade/Gas_natural (and other state-priced
    fuels) use Fuel_Prices_State when available, else the national price.
    Charcoal is EXCLUDED (endogenous national 3-tier curve). The ore/pellet
    state differential (ore_delta) is added per (route, state).
    """
    cost = {}
    for r in cfg["routes"]:
        for y in cfg["YEARS"]:
            total = 0.0
            for f in cfg["fuels_by_route"][r]:
                if f == CHARCOAL_FUEL:
                    continue  # endogenous — handled separately
                if f in STATE_PRICED_FUELS and (f, state) in cfg["prices_state"]:
                    price = cfg["prices_state"][(f, state)]
                else:
                    price = cfg["prices"].get((f, y), 0.0)
                total += cfg["ei"].get((r, f), 0.0) * price
            # ore/pellet state differential (USD/t), no year dimension
            total += cfg.get("ore_delta", {}).get((r, state), 0.0)
            cost[(r, y)] = total
    return cost


def compute_route_fuel_cost_by_state(cfg: dict, states) -> dict:
    """LOC: {(route, year, state): cost_USD_per_t} for every state in `states`."""
    out = {}
    for state in states:
        per_state = compute_route_fuel_cost_state(cfg, state)
        for (r, y), c in per_state.items():
            out[(r, y, state)] = c
    return out


def compute_route_charcoal_use(cfg: dict) -> dict:
    """charcoal_use[r] = EI[r, CHARCOAL_FUEL]  (GJ of charcoal per t of steel).

    This is the per-tonne charcoal intensity of each route, used both to
    build the yearly charcoal-demand expression and (×1000) the GJ totals.
    """
    use = {}
    for r in cfg["routes"]:
        use[r] = cfg["ei"].get((r, CHARCOAL_FUEL), 0.0)
    return use


def compute_route_biomass_use(cfg: dict) -> dict:
    """biomass_use[r] = Σ_fuel_in_biomass  EI[r,f]   (GJ per t)
    Currently treats 'Carvao_vegetal' (sanitized) as the only biomass fuel.
    """
    BIOMASS_FUELS = {"Carvao_vegetal"}   # underscore — matches sanitized cfg
    use = {}
    for r in cfg["routes"]:
        total = 0.0
        for f in cfg["fuels_by_route"][r]:
            if f in BIOMASS_FUELS:
                total += cfg["ei"].get((r, f), 0.0)
        use[r] = total
    return use


def compute_route_total_energy(cfg: dict) -> dict:
    """total_energy_use[r] = Σ_fuel_classified  EI[r,f]   (GJ per t of steel).

    Only fuels present in FUEL_CATEGORY are counted (mirrors the reporting
    layer's energy-mix calculation, which also drops unclassified fuels).
    Used as the denominator of the fossil-fuel-share constraint (CONSTRAINT I).
    """
    use = {}
    for r in cfg["routes"]:
        total = 0.0
        for f in cfg["fuels_by_route"][r]:
            if f in FUEL_CATEGORY:
                total += cfg["ei"].get((r, f), 0.0)
        use[r] = total
    return use


def compute_route_fossil_energy(cfg: dict) -> dict:
    """fossil_energy_use[r] = Σ_fuel_fossil  EI[r,f]   (GJ per t of steel).

    Only fuels classified as "Fossil" in FUEL_CATEGORY are counted.
    Used as the numerator of the fossil-fuel-share constraint (CONSTRAINT I).
    """
    use = {}
    for r in cfg["routes"]:
        total = 0.0
        for f in cfg["fuels_by_route"][r]:
            if FUEL_CATEGORY.get(f) == "Fossil":
                total += cfg["ei"].get((r, f), 0.0)
        use[r] = total
    return use


def build_charcoal_supply_tiers():
    """Build the 3-tier charcoal supply curve.

    Reads CHARCOAL_TIER_WIDTH_PJ and CHARCOAL_TIER_PRICE from the config
    block at the top of this file and converts widths to GJ (the unit the
    model uses internally for charcoal demand).

    Returns
    -------
    seg_width_GJ : list[float]   width of each tier in GJ/year
    seg_price    : list[float]   marginal price of each tier in USD/GJ
                                 (cost contribution of tier s = price[s] * delta[s])

    The list of tiers is the same for every year — the supply curve does
    not depend on time. The model only needs to multiply each tier-usage
    variable by its price and sum.
    """
    widths_PJ = list(CHARCOAL_TIER_WIDTH_PJ)
    prices    = list(CHARCOAL_TIER_PRICE)

    if len(widths_PJ) != len(prices):
        raise ValueError("CHARCOAL_TIER_WIDTH_PJ and CHARCOAL_TIER_PRICE "
                         "must have the same length.")
    if any(w <= 0 for w in widths_PJ):
        raise ValueError("All CHARCOAL_TIER_WIDTH_PJ entries must be positive.")
    # Convexity check: prices must be non-decreasing so the cheap-first
    # selection by the solver is automatic without binaries.
    for i in range(1, len(prices)):
        if prices[i] < prices[i - 1] - 1e-9:
            raise ValueError(
                f"CHARCOAL_TIER_PRICE must be non-decreasing for the "
                f"no-binary linearization to work; got {prices}."
            )

    if CHARCOAL_ABOVE_CAP == "extend":
        widths_PJ.append(float(CHARCOAL_EXTEND_WIDTH_PJ))
        prices.append(prices[-1])
    elif CHARCOAL_ABOVE_CAP != "hard_cap":
        raise ValueError(f"CHARCOAL_ABOVE_CAP must be 'hard_cap' or 'extend', "
                         f"got {CHARCOAL_ABOVE_CAP!r}.")

    seg_width_GJ = [w * 1e6 for w in widths_PJ]   # PJ -> GJ
    seg_price = prices
    return seg_width_GJ, seg_price


def penetration_fraction(year: int, year_start: int, year_end: int) -> float:
    """Smoothstep sigmoid for technology penetration, exact at the endpoints.

        y <= year_start  -> 0
        y >= year_end    -> 1
        in between       -> 3 t^2 - 2 t^3,  t = (y - year_start) / (year_end - year_start)

    The cubic smoothstep has derivative zero at both ends (slow entry, slow
    saturation) and hits 0 and 1 exactly at the boundary years — unlike a
    logistic, which is only asymptotic.
    """
    if year_end <= year_start:
        raise ValueError(f"penetration window must satisfy end > start; "
                         f"got start={year_start}, end={year_end}.")
    if year <= year_start:
        return 0.0
    if year >= year_end:
        return 1.0
    t = (year - year_start) / (year_end - year_start)
    return 3.0 * t * t - 2.0 * t * t * t


# ---- Plant size menu (kt) per technology route ----
SIZE_MENU = {
    "BF-BOF_MC":  [2000, 4000, 6000],
    "BF-BOF_CC":  [1500, 2500, 4000],
    "BF-BOF-CCS": [2000, 4000, 6000],
    "EAF":        [500, 1000, 2000],
    "DR-NG":      [1000, 2000, 3000],
    "DR-H2":      [500, 1500, 2500],
    "IBT":        [1000, 2000, 3000],
    "DR-NG-CCS": [1000, 2000, 3000],   # NEW — mirrors DR-NG sizing; adjust if the real plant scale differs
}
MAX_PLANTS_PER_SIZE = 5


#==============================================================================
# 5. BUILD AND SOLVE THE OPTIMIZATION MODEL
#==============================================================================

def build_model(plants: pd.DataFrame, cfg: dict) -> ConcreteModel:
    """
    Decision variables
    ------------------
    1. production_existing[plantID, year]
    2. production_succ[old_plantID, route, year]
    3. production_greenfield[route, state, year]        (state dim, LOC)
    4. active_succ[old_plantID, route, year]  ∈ {0, 1}
    5. n_green[route, size, state, year]      ∈ ℤ+       (state dim, LOC)
    6. charcoal_delta[year, segment]          ∈ ℝ+   (NEW, V20_8)

    Endogenous charcoal price (NEW, V20_8)
    --------------------------------------
    Yearly charcoal demand D[y] (GJ) is a linear expression of production.
    Its total cost C(D) = k·D² (convex) is approximated by linear segments:
        D[y]      = Σ_s charcoal_delta[y, s]
        cost(y)   = Σ_s seg_slope[s] · charcoal_delta[y, s]
        0 ≤ charcoal_delta[y, s] ≤ seg_width[s]
    Because C is convex and the model minimizes, the solver fills the
    cheaper (lower-slope) segments first — no binaries needed.

    Note on CCS (V20_12): any route listed in CCS_ROUTES (currently
    BF-BOF-CCS and DR-NG-CCS) has its emission factor discounted by
    CAPTURE_RATE_CCS inside compute_route_emission_factor(). No other
    model mechanics need to know which routes use CCS — the emission
    cap, cost accounting, and reporting all key off ef_route, which
    already reflects the capture.
    """
    m = ConcreteModel()

    m.PLANTS = Set(initialize=plants["PlantID"].tolist())
    m.ROUTES = Set(initialize=cfg["routes"])
    m.YEARS  = Set(initialize=cfg["YEARS"])

    plant_info = plants.set_index("PlantID").to_dict("index")

    def is_plant_active(p, y):
        info = plant_info[p]
        return info["Startyear"] <= y <= info["Retrofitdate"]

    L           = cfg["PLANT_LIFETIME"]
    cap_mult    = cfg["CAPACITY_MULTIPLIER"]
    year_end    = cfg["YEAR_END"]
    year_start  = cfg["YEAR_START"]
    ramp_floor  = 1.0 - cfg["MAX_RAMP_DOWN"]
    min_util    = cfg["MIN_UTILIZATION"]

    # Valid (route, size) combinations
    route_size_pairs = [(r, sz) for r in SIZE_MENU for sz in SIZE_MENU[r]]
    m.ROUTE_SIZE = Set(initialize=route_size_pairs, dimen=2)

    # LOC ------------------------------------------------------------------
    # STATE / LOCATION SETUP
    # ----------------------------------------------------------------------
    # Candidate greenfield states: from the Greenfield_States sheet, else all
    # 27 UFs, subject to the per-route geographic restrictions below.
    if cfg.get("greenfield_states"):
        candidate_states = sorted(set(cfg["greenfield_states"]))
    else:
        candidate_states = sorted(ALL_BR_STATES)
        print(f"    [info] No 'Greenfield_States' sheet — defaulting greenfield "
              f"candidate states to ALL {len(candidate_states)} Brazilian UFs.")

    def allowed_states_for_route(r):
        restriction = ROUTE_STATE_RESTRICTIONS.get(r)
        if restriction is None:
            return set(candidate_states)
        return set(candidate_states) & set(restriction)

    route_allowed_states = {r: allowed_states_for_route(r) for r in cfg["routes"]}
    for r, states_ok in route_allowed_states.items():
        if r in ROUTE_STATE_RESTRICTIONS and not states_ok:
            print(f"    [warn] Route {r!r} has a geographic restriction but NO "
                  f"candidate state satisfies it — it will get zero NEW capacity.")

    m.STATES = Set(initialize=candidate_states)

    route_state_pairs = [
        (r, s) for r in cfg["routes"] for s in route_allowed_states[r]
    ]
    m.ROUTE_STATE = Set(initialize=route_state_pairs, dimen=2)

    route_size_state_triples = [
        (r, sz, s) for r in SIZE_MENU for sz in SIZE_MENU[r]
        for s in route_allowed_states.get(r, set())
    ]
    m.ROUTE_SIZE_STATE = Set(initialize=route_size_state_triples, dimen=3)

    # Precompute state-specific fuel cost for every candidate state plus every
    # state that actually hosts an existing plant.
    all_relevant_states = set(candidate_states) | set(plants["UF"].unique().tolist())
    cost_route_by_state = compute_route_fuel_cost_by_state(cfg, all_relevant_states)
    # LOC end --------------------------------------------------------------

    slot_info = {}
    for p in plants["PlantID"]:
        info = plant_info[p]
        retire = info["Retrofitdate"]
        if retire < year_end:
            slot_start = retire + 1
            slot_end   = min(retire + L, year_end)
            slot_info[p] = {
                "start_year":        slot_start,
                "end_year":          slot_end,
                "slot_capacity":     info["Capacity"] * cap_mult,
                "original_route":    info["Route"],
                "original_capacity": info["Capacity"],
                "state":             info["UF"],          # LOC
            }

    print(f"    {len(slot_info)} successor slots created "
          f"(plants whose Retrofitdate < {year_end}).")

    m.SLOTS = Set(initialize=list(slot_info.keys()))

    def is_slot_active(p_old, y):
        s = slot_info[p_old]
        return s["start_year"] <= y <= s["end_year"]

    def slot_route_allowed(p_old, r):  # LOC
        """Whether route r may be installed at slot p_old, given the slot's
        fixed physical state and r's geographic restriction (if any)."""
        restriction = ROUTE_STATE_RESTRICTIONS.get(r)
        if restriction is None:
            return True
        return slot_info[p_old]["state"] in restriction

    # ---- Continuous production variables
    m.production_existing   = Var(m.PLANTS, m.YEARS, domain=NonNegativeReals)
    m.production_succ       = Var(m.SLOTS, m.ROUTES, m.YEARS, domain=NonNegativeReals)
    m.production_greenfield = Var(m.ROUTE_STATE, m.YEARS, domain=NonNegativeReals)  # LOC: +state

    # ---- Greenfield: integer count of plants per (route, size, state), per year
    m.n_green = Var(m.ROUTE_SIZE_STATE, m.YEARS,  # LOC: +state
                    domain=NonNegativeIntegers, bounds=(0, MAX_PLANTS_PER_SIZE))

    # ---- Successor: binary active per (slot, route, year)
    m.active_succ = Var(m.SLOTS, m.ROUTES, m.YEARS, domain=Binary)

    # ---- Vintage auxiliary variables for finite-window CAPEX amortization
    # (NEW) — "positive part" of the built-in-the-last-L-years capacity, used
    # instead of a raw telescoping difference because active_succ can be
    # forced back to 0 when a slot's window closes (see A2.0), which would
    # otherwise make the raw difference go negative (a bogus CAPEX credit).
    m.capex_vintage_succ  = Var(m.SLOTS, m.ROUTES, m.YEARS, domain=NonNegativeReals)
    m.capex_vintage_green = Var(m.ROUTE_SIZE_STATE, m.YEARS, domain=NonNegativeReals)  # LOC: +state

    # ---- Charcoal supply tiers (3 cumulative tiers, same every year)
    seg_width, seg_price = build_charcoal_supply_tiers()
    n_seg = len(seg_width)
    m.CHARCOAL_SEG = Set(initialize=list(range(n_seg)))
    # delta[y, s] = how many GJ of tier s are used in year y, bounded by width.
    m.charcoal_delta = Var(
        m.YEARS, m.CHARCOAL_SEG, domain=NonNegativeReals,
        bounds=lambda mm, y, s: (0.0, seg_width[s]),
    )

    total_supply_PJ = sum(seg_width) / 1e6
    print(f"    Charcoal supply: {n_seg} tier(s), total supply "
          f"{total_supply_PJ:.1f} PJ/year ({CHARCOAL_ABOVE_CAP}).")
    for s, (w, p) in enumerate(zip(seg_width, seg_price), start=1):
        print(f"      tier {s}: {w/1e6:>6.1f} PJ/year at {p:5.2f} USD/GJ")

    ef_route       = compute_route_emission_factor(cfg)
    # LOC: fuel cost is no longer a single national dict — it depends on WHERE
    # a unit is located. cost_route_by_state[(r, y, state)] gives the USD/t.
    biom_route     = compute_route_biomass_use(cfg)
    charcoal_route = compute_route_charcoal_use(cfg)     # NEW
    total_energy_route  = compute_route_total_energy(cfg)   # NEW (fossil-share constraint)
    fossil_energy_route = compute_route_fossil_energy(cfg)  # NEW (fossil-share constraint)

    ccs_routes_in_model = sorted(r for r in CCS_ROUTES if r in cfg["routes"])
    print(f"    CCS routes active in this config: {ccs_routes_in_model}")

    def existing_fuel_cost(p, y):  # LOC
        r = plant_info[p]["Route"]; st = plant_info[p]["UF"]
        return cost_route_by_state.get((r, y, st), 0.0)

    def succ_fuel_cost(p_old, r, y):  # LOC
        st = slot_info[p_old]["state"]
        return cost_route_by_state.get((r, y, st), 0.0)

    # ---- Helper: total greenfield installed capacity from size-menu variables
    def green_installed_cap(r, st, y):  # LOC: +state
        """Total greenfield installed capacity of route r in state st, year y (kt)."""
        return sum(m.n_green[r, sz, st, y] * sz for sz in SIZE_MENU[r]
                   if (r, sz, st) in m.ROUTE_SIZE_STATE)

    # ---- Helper: total charcoal demand in year y (GJ) as a linear expression
    def charcoal_demand_expr(y):
        """Σ over all production of (production_kt · 1000 · EI_charcoal[route])."""
        dem = 0.0
        for p in m.PLANTS:
            r = plant_info[p]["Route"]
            dem += m.production_existing[p, y] * 1000 * charcoal_route[r]
        for p_old in m.SLOTS:
            for r in m.ROUTES:
                dem += m.production_succ[p_old, r, y] * 1000 * charcoal_route[r]
        for (r, st) in m.ROUTE_STATE:  # LOC: +state
            dem += m.production_greenfield[r, st, y] * 1000 * charcoal_route[r]
        return dem

    # ========================================================================
    # CONSTRAINT A — Capacity of existing plants (always active in window)
    # ========================================================================
    def cap_existing_upper(m, p, y):
        info = plant_info[p]
        if not is_plant_active(p, y):
            return m.production_existing[p, y] == 0
        return m.production_existing[p, y] <= info["Capacity"]
    m.C_capacity_existing = Constraint(m.PLANTS, m.YEARS, rule=cap_existing_upper)

    def cap_existing_lower(m, p, y):
        info = plant_info[p]
        if not is_plant_active(p, y):
            return Constraint.Skip
        return m.production_existing[p, y] >= info["Capacity"] * min_util
    m.C_min_utilization = Constraint(m.PLANTS, m.YEARS, rule=cap_existing_lower)

    # ========================================================================
    # CONSTRAINT A2 — Successor slots: capacity + binary coupling per (slot, route)
    # ========================================================================

    # A2.0: Force active_succ = 0 outside slot window — for every route
    def succ_active_window_rule(m, p_old, r, y):
        if not is_slot_active(p_old, y):
            return m.active_succ[p_old, r, y] == 0
        return Constraint.Skip
    m.C_succ_active_window = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                        rule=succ_active_window_rule)

    # A2.1: Production upper bound = slot_cap × active_succ
    def cap_succ_upper(m, p_old, r, y):
        cap = slot_info[p_old]["slot_capacity"]
        return m.production_succ[p_old, r, y] <= cap * m.active_succ[p_old, r, y]
    m.C_capacity_succ_upper = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                         rule=cap_succ_upper)

    # A2.2: Min utilization when active
    def cap_succ_lower(m, p_old, r, y):
        cap = slot_info[p_old]["slot_capacity"]
        return (m.production_succ[p_old, r, y]
                >= min_util * cap * m.active_succ[p_old, r, y])
    m.C_capacity_succ_lower = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                         rule=cap_succ_lower)

    # A2.3: Site limit — sum of routes in same slot ≤ slot_capacity
    def cap_succ_aggregate(m, p_old, y):
        if not is_slot_active(p_old, y):
            return Constraint.Skip
        cap = slot_info[p_old]["slot_capacity"]
        return sum(m.production_succ[p_old, r, y] for r in m.ROUTES) <= cap
    m.C_capacity_succ_aggregate = Constraint(m.SLOTS, m.YEARS,
                                             rule=cap_succ_aggregate)

    # A2.4: Monotonicity per (slot, route): once active, stays active
    def succ_monotone_rule(m, p_old, r, y):
        s = slot_info[p_old]
        if y <= s["start_year"]:
            return Constraint.Skip
        if y > s["end_year"]:
            return Constraint.Skip
        return m.active_succ[p_old, r, y] >= m.active_succ[p_old, r, y - 1]
    m.C_succ_monotone = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                   rule=succ_monotone_rule)

    # A2.5 (NEW): Minimum 25-year (= PLANT_LIFETIME) commitment once built.
    # The slot window is fixed at [Retrofitdate+1, Retrofitdate+L] (or
    # truncated by YEAR_END), so the ONLY year that leaves a full L years
    # before the window closes is the window's very first year
    # (start_year). Any activation after "cutoff" = end_year - L + 1 would
    # be forced to shut down before reaching L years when the window
    # closes — so we forbid NEW activation past that cutoff. Combined
    # with the existing monotonicity (A2.4) and the hard window-close
    # (A2.0), this means each slot either (a) is never used at all, or
    # (b) is built exactly at start_year and runs the full window.
    # If the window itself is shorter than L (truncated by YEAR_END —
    # i.e. the original plant retires too close to the horizon's end to
    # ever host a full 25-year successor), cutoff < start_year and this
    # constraint forces the slot to stay at 0 for its entire window —
    # it becomes permanently unusable.
    def succ_min_duration_rule(m, p_old, r, y):
        s = slot_info[p_old]
        if y < s["start_year"] or y > s["end_year"]:
            return Constraint.Skip
        cutoff = s["end_year"] - L + 1
        if y <= cutoff:
            return Constraint.Skip   # building here still allows a full L-year run
        prev = m.active_succ[p_old, r, y - 1] if y - 1 >= s["start_year"] else 0
        return m.active_succ[p_old, r, y] == prev   # no NEW activation past cutoff
    m.C_succ_min_duration = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                        rule=succ_min_duration_rule)

    # ========================================================================
    # CONSTRAINT A3 — Greenfield: size-menu capacity coupling + monotonicity
    # ========================================================================

    # A3.1: Production upper bound = installed capacity   (LOC: per route+state)
    def cap_green_upper(m, r, st, y):
        return m.production_greenfield[r, st, y] <= green_installed_cap(r, st, y)
    m.C_capacity_green_upper = Constraint(m.ROUTE_STATE, m.YEARS,
                                          rule=cap_green_upper)

    # A3.2: Min utilization on installed capacity   (LOC: per route+state)
    def cap_green_lower(m, r, st, y):
        return (m.production_greenfield[r, st, y]
                >= min_util * green_installed_cap(r, st, y))
    m.C_capacity_green_lower = Constraint(m.ROUTE_STATE, m.YEARS,
                                          rule=cap_green_lower)

    # A3.3: Monotonicity — can't demolish, per (route, size, state)   (LOC)
    def green_monotone_rule(m, r, sz, st, y):
        if y == year_start:
            return Constraint.Skip
        return m.n_green[r, sz, st, y] >= m.n_green[r, sz, st, y - 1]
    m.C_green_monotone = Constraint(m.ROUTE_SIZE_STATE, m.YEARS,
                                    rule=green_monotone_rule)

    # NOTE: a symmetric "must-run-25-years" cutoff for greenfield builds
    # (forbidding new n_green after YEAR_END - L + 1) was tested here and
    # REMOVED — combined with the successor-slot rule above (A2.5) it made
    # the model infeasible in stress testing: both investment channels
    # would be frozen after ~2046, leaving no feasible way to keep meeting
    # a production target that keeps growing all the way to YEAR_END while
    # also respecting the technology-penetration ramp-up curves. Greenfield
    # was never subject to the "1-year blip" problem this constraint set
    # out to fix (it has no forced decommissioning), so leaving it
    # unconstrained here is safe and doesn't reopen that issue.

    # ========================================================================
    # CONSTRAINT B — Production target
    # ========================================================================
    def target_rule(m, y):
        existing = sum(m.production_existing[p, y] for p in m.PLANTS)
        succ     = sum(m.production_succ[p_old, r, y]
                       for p_old in m.SLOTS for r in m.ROUTES)
        green    = sum(m.production_greenfield[r, st, y] for (r, st) in m.ROUTE_STATE)  # LOC
        return existing + succ + green == cfg["production_target"][y]
    m.C_production_target = Constraint(m.YEARS, rule=target_rule)

    # ========================================================================
    # CONSTRAINT C — Emission cap
    # ========================================================================
    def emission_rule(m, y):
        emis_existing = sum(
            m.production_existing[p, y] * 1000 * ef_route[plant_info[p]["Route"]]
            for p in m.PLANTS
        )
        emis_succ = sum(
            m.production_succ[p_old, r, y] * 1000 * ef_route[r]
            for p_old in m.SLOTS for r in m.ROUTES
        )
        emis_green = sum(
            m.production_greenfield[r, st, y] * 1000 * ef_route[r]
            for (r, st) in m.ROUTE_STATE  # LOC
        )
        return emis_existing + emis_succ + emis_green <= cfg["emission_cap"][y]
    m.C_emission_cap = Constraint(m.YEARS, rule=emission_rule)

    # ========================================================================
    # CONSTRAINT D — Scrap supply
    # ========================================================================
    # LOC: the single national cap is split into one constraint per macro-
    # region g. For each region and year y:
    #   LHS = production * scrap_rate over units whose UF belongs to g
    #   RHS = national scrap_supply[y] * regional share[g]
    # UF of each unit: existing -> plant_info[p]["UF"]; successor ->
    # slot_info[p_old]["state"]; greenfield -> st. Rigid boundary (no flow).
    m.REGIONS = Set(initialize=sorted(cfg["scrap_share"].keys()))

    def scrap_rule(m, g, y):
        scrap_existing = sum(
            m.production_existing[p, y] * cfg["scrap_rate"][plant_info[p]["Route"]]
            for p in m.PLANTS
            if cfg["uf_region"][plant_info[p]["UF"]] == g
        )
        scrap_succ = sum(
            m.production_succ[p_old, r, y] * cfg["scrap_rate"][r]
            for p_old in m.SLOTS for r in m.ROUTES
            if cfg["uf_region"][slot_info[p_old]["state"]] == g
        )
        scrap_green = sum(
            m.production_greenfield[r, st, y] * cfg["scrap_rate"][r]
            for (r, st) in m.ROUTE_STATE
            if cfg["uf_region"][st] == g
        )
        return (scrap_existing + scrap_succ + scrap_green
                <= cfg["scrap_supply"][y] * cfg["scrap_share"][g])
    m.C_scrap_supply = Constraint(m.REGIONS, m.YEARS, rule=scrap_rule)

    # ========================================================================
    # CONSTRAINT E — Biomass supply
    # ========================================================================
    def biomass_rule(m, y):
        biom_existing = sum(
            m.production_existing[p, y] * 1000 * biom_route[plant_info[p]["Route"]]
            for p in m.PLANTS
        )
        biom_succ = sum(
            m.production_succ[p_old, r, y] * 1000 * biom_route[r]
            for p_old in m.SLOTS for r in m.ROUTES
        )
        biom_green = sum(
            m.production_greenfield[r, st, y] * 1000 * biom_route[r]
            for (r, st) in m.ROUTE_STATE  # LOC
        )
        return biom_existing + biom_succ + biom_green <= cfg["biomass_supply"][y]
    m.C_biomass_supply = Constraint(m.YEARS, rule=biomass_rule)

    # ========================================================================
    # CONSTRAINT F (NEW, V20_8) — Charcoal demand = sum of cost-curve segments
    # ========================================================================
    # Ties the linear charcoal-demand expression to the segment variables.
    # The objective then pays Σ_s seg_slope[s]·charcoal_delta[y,s] for it.
    def charcoal_balance_rule(m, y):
        return charcoal_demand_expr(y) == sum(
            m.charcoal_delta[y, s] for s in m.CHARCOAL_SEG
        )
    m.C_charcoal_balance = Constraint(m.YEARS, rule=charcoal_balance_rule)

    # ========================================================================
    # CONSTRAINT H — Technology penetration limit (CCS, H2)
    # ========================================================================
    # For each entry in PENETRATION_LIMITS, the combined production of the
    # listed routes is capped at s(y) * production_target[y], where s(y) is
    # a smoothstep that ramps from 0 to 1 between the given start and end
    # years. Existing plants are not affected (none of them are CCS/H2 in
    # this config) — the cap applies to successor + greenfield production.
    #
    # Keys in PENETRATION_LIMITS can be a single route (str) or a tuple of
    # routes (joint cap). Routes not present in cfg["routes"] are skipped
    # with a warning.

    def _normalize_pen_key(key):
        return (key,) if isinstance(key, str) else tuple(key)

    pen_groups = []
    for key, params_ in PENETRATION_LIMITS.items():
        routes_in_group = _normalize_pen_key(key)
        # validate routes exist in this config
        unknown = [r for r in routes_in_group if r not in cfg["routes"]]
        if unknown:
            print(f"    [warn] PENETRATION_LIMITS key {key!r}: route(s) "
                  f"{unknown} not in cfg['routes']; entry skipped.")
            continue
        ys = int(params_["start"]); ye = int(params_["end"])
        pen_groups.append((routes_in_group, ys, ye))
        print(f"    Penetration limit on {routes_in_group}: "
              f"s({ys})=0 -> s({ye})=1 (smoothstep).")

    if pen_groups:
        # index each constraint by (group_idx, year)
        m.PEN_GROUP_IDX = Set(initialize=list(range(len(pen_groups))))

        def penetration_rule(m, g, y):
            routes_in_group, ys, ye = pen_groups[g]
            s = penetration_fraction(y, ys, ye)
            target = cfg["production_target"][y]
            prod_group = sum(
                m.production_succ[p_old, r, y]
                for p_old in m.SLOTS for r in routes_in_group
            ) + sum(
                m.production_greenfield[r, st, y]
                for (r, st) in m.ROUTE_STATE if r in routes_in_group  # LOC
            )
            return prod_group <= s * target
        m.C_penetration = Constraint(m.PEN_GROUP_IDX, m.YEARS,
                                     rule=penetration_rule)

    # store for the report
    m._pen_groups = pen_groups

    # ========================================================================
    # CONSTRAINT I (NEW) — Maximum fossil-fuel share of the energy mix
    # ========================================================================
    # Caps the aggregate fossil-fuel energy (GJ) — summed over ALL production
    # (existing + successors + greenfield) — at a fraction of total energy
    # use (GJ) in that year:
    #
    #     fossil_energy[y]  <=  fossil_share_max[y] * total_energy[y]
    #
    # "Fossil" and "total" follow the same FUEL_CATEGORY classification used
    # by the reporting layer (Energy_mix sheet), so the constraint and the
    # post-hoc indicator are always consistent. fossil_share_max[y] is a
    # fixed parameter (from the Fossil_Share_Max sheet), so the constraint
    # is linear — no new binaries needed.
    def fossil_share_rule(m, y):
        fossil_existing = sum(
            m.production_existing[p, y] * 1000 * fossil_energy_route[plant_info[p]["Route"]]
            for p in m.PLANTS
        )
        fossil_succ = sum(
            m.production_succ[p_old, r, y] * 1000 * fossil_energy_route[r]
            for p_old in m.SLOTS for r in m.ROUTES
        )
        fossil_green = sum(
            m.production_greenfield[r, st, y] * 1000 * fossil_energy_route[r]
            for (r, st) in m.ROUTE_STATE  # LOC
        )
        fossil_total = fossil_existing + fossil_succ + fossil_green

        total_existing = sum(
            m.production_existing[p, y] * 1000 * total_energy_route[plant_info[p]["Route"]]
            for p in m.PLANTS
        )
        total_succ = sum(
            m.production_succ[p_old, r, y] * 1000 * total_energy_route[r]
            for p_old in m.SLOTS for r in m.ROUTES
        )
        total_green = sum(
            m.production_greenfield[r, st, y] * 1000 * total_energy_route[r]
            for (r, st) in m.ROUTE_STATE  # LOC
        )
        energy_total = total_existing + total_succ + total_green

        return fossil_total <= cfg["fossil_share_max"][y] * energy_total
    m.C_fossil_share = Constraint(m.YEARS, rule=fossil_share_rule)

    # ========================================================================
    # CONSTRAINT G — Ramp-down (max -10% YoY drop within a "platau")
    # ========================================================================

    # G1: Existing plants — per plant
    def ramp_existing_rule(m, p, y):
        if y == year_start:
            return Constraint.Skip
        info = plant_info[p]
        if not is_plant_active(p, y - 1):
            return Constraint.Skip
        if y >= info["Retrofitdate"]:
            return Constraint.Skip
        return m.production_existing[p, y] >= ramp_floor * m.production_existing[p, y - 1]
    m.C_ramp_existing = Constraint(m.PLANTS, m.YEARS, rule=ramp_existing_rule)

    # G2: Successor slots — per (slot, route)
    def ramp_succ_rule(m, p_old, r, y):
        if y == year_start:
            return Constraint.Skip
        s = slot_info[p_old]
        if not (s["start_year"] <= y - 1 <= s["end_year"]):
            return Constraint.Skip
        if y >= s["end_year"]:
            return Constraint.Skip
        if not is_slot_active(p_old, y):
            return Constraint.Skip
        return m.production_succ[p_old, r, y] \
               >= ramp_floor * m.production_succ[p_old, r, y - 1]
    m.C_ramp_succ = Constraint(m.SLOTS, m.ROUTES, m.YEARS, rule=ramp_succ_rule)

    # G3: Greenfield — per (route, state). No end-of-life skip.   (LOC)
    def ramp_green_rule(m, r, st, y):
        if y == year_start:
            return Constraint.Skip
        return (m.production_greenfield[r, st, y]
                >= ramp_floor * m.production_greenfield[r, st, y - 1])
    m.C_ramp_greenfield = Constraint(m.ROUTE_STATE, m.YEARS, rule=ramp_green_rule)

    # ========================================================================
    # OBJECTIVE — minimize discounted total cost
    # ========================================================================
    # Retrofit (successor slots) is cheaper than greenfield (new plant) —
    # each pathway is annualized from its own CAPEX column.
    #
    # CHANGED: CAPEX is now annualized with the LEVELIZED (capital-recovery-
    # factor) formula instead of straight-line division. CRF converts a
    # lump-sum CAPEX into a constant annual payment over L years such that
    # its present value (discounted at DISCOUNT_RATE) exactly equals the
    # original CAPEX — the standard approach behind LCOE/LCOS:
    #
    #     CRF = r(1+r)^L / [(1+r)^L - 1]        (r = DISCOUNT_RATE, L = PLANT_LIFETIME)
    #     CAPEX_annual = CAPEX_total * CRF
    #
    # This replaces the old CAPEX_total / L straight-line rule. CRF > 1/L
    # whenever r > 0 (it embeds the cost of capital), so this raises the
    # annual CAPEX charge relative to the previous version. The L-year
    # vintage window (capex_vintage_succ / capex_vintage_green, further
    # below) is unchanged and still amortizes over exactly L years — CRF
    # is defined for that same L, so the pairing is consistent.
    r_disc = cfg["DISCOUNT_RATE"]
    if r_disc == 0:
        CRF = 1.0 / L   # degenerate case: no cost of capital, falls back to straight-line
    else:
        CRF = r_disc * (1 + r_disc) ** L / ((1 + r_disc) ** L - 1)

    capex_annual_retrofit   = {r: cfg["capex_retrofit"][r]   * CRF for r in cfg["routes"]}
    capex_annual_greenfield = {r: cfg["capex_greenfield"][r] * CRF for r in cfg["routes"]}

    # ------------------------------------------------------------------
    # CHANGED: the retrofit CAPEX rate only applies when a successor slot
    # rebuilds the SAME route the original plant had (a genuine "revamp in
    # place"). Switching to a different route at that site is NOT a
    # retrofit in the CAPEX sense — the site still needs the full new
    # process equipment — so it is charged the GREENFIELD rate instead.
    # Only the land/utilities/logistics saving (implicit in the lower
    # retrofit column) is retrofit-specific, and that only holds for a
    # like-for-like rebuild.
    # ------------------------------------------------------------------
    def capex_annual_succ(p_old, r):
        same_route = (r == slot_info[p_old]["original_route"])
        return capex_annual_retrofit[r] if same_route else capex_annual_greenfield[r]

    # ------------------------------------------------------------------
    # CHANGED: CAPEX is now amortized over a FINITE window of L years
    # (L = PLANT_LIFETIME), instead of being charged every year from
    # construction through the end of the horizon.
    #
    # Naively, "capacity built within the last L years" for a monotonic
    # 0/1 (or integer count) series would be the telescoping difference
    # value(y) - value(y - L). But active_succ can be forced back to 0
    # when a slot's window closes (Constraint A2.0) — NOT just when it's
    # built — so that raw difference can go NEGATIVE (a bogus CAPEX
    # credit) once a slot's window ends. We instead bound an auxiliary
    # non-negative variable from below by that difference; since CAPEX
    # cost is strictly positive and the variable appears nowhere else,
    # the minimizing solver drives it to exactly max(0, difference) —
    # the standard LP "positive part" trick. No new binaries needed.
    # ------------------------------------------------------------------
    def active_succ_vt(p_old, r, y):
        """active_succ[p_old, r, y], treated as 0 for y before YEAR_START."""
        return m.active_succ[p_old, r, y] if y >= year_start else 0

    def n_green_vt(r, sz, st, y):  # LOC: +state
        """n_green[r, sz, st, y], treated as 0 for y before YEAR_START."""
        return m.n_green[r, sz, st, y] if y >= year_start else 0

    def capex_vintage_succ_rule(m, p_old, r, y):
        return (m.capex_vintage_succ[p_old, r, y]
                >= m.active_succ[p_old, r, y] - active_succ_vt(p_old, r, y - L))
    m.C_capex_vintage_succ = Constraint(m.SLOTS, m.ROUTES, m.YEARS,
                                         rule=capex_vintage_succ_rule)

    def capex_vintage_green_rule(m, r, sz, st, y):  # LOC: +state
        return (m.capex_vintage_green[r, sz, st, y]
                >= m.n_green[r, sz, st, y] - n_green_vt(r, sz, st, y - L))
    m.C_capex_vintage_green = Constraint(m.ROUTE_SIZE_STATE, m.YEARS,
                                          rule=capex_vintage_green_rule)

    def annual_cost(y):
        discount = (1 + cfg["DISCOUNT_RATE"]) ** (y - cfg["YEAR_START"])

        # --- EXISTING: CAPEX (CHANGED — see below) + OPEX_fixed on capacity,
        #     fuel on production (cost_route excludes charcoal — added below)
        #
        # CHANGED: existing plants now ALSO carry a CRF-annualized CAPEX
        # charge, using their own route's CAPEX_retrofit rate (capex_annual_
        # retrofit, already levelized by CRF above) — no longer treated as
        # pure sunk cost. Unlike the successor/greenfield vintage logic,
        # there is NO L-year cutoff here: the charge applies every year the
        # plant is active, from Startyear through Retrofitdate, however long
        # that span is (per explicit instruction — this deliberately departs
        # from the "sunk cost" convention used elsewhere in the objective).
        c_exist_fixed = 0.0
        c_exist_var   = 0.0
        for p in m.PLANTS:
            r = plant_info[p]["Route"]
            cap_p = plant_info[p]["Capacity"]
            if plant_info[p]["Startyear"] <= y <= plant_info[p]["Retrofitdate"]:
                c_exist_fixed += cap_p * 1000 * (cfg["opex_fixed"][r] + capex_annual_retrofit[r])
            c_exist_var += m.production_existing[p, y] * 1000 * existing_fuel_cost(p, y)  # LOC

        # --- SUCCESSOR: CAPEX amortized over L years (vintage window),
        #     OPEX_fixed charged every year the slot is active, fuel on production
        c_succ_fixed = 0.0
        c_succ_var   = 0.0
        for p_old in m.SLOTS:
            s_cap = slot_info[p_old]["slot_capacity"]
            for r in m.ROUTES:
                inst_cap = s_cap * m.active_succ[p_old, r, y]   # for OPEX
                capex_charge = m.capex_vintage_succ[p_old, r, y] * s_cap
                c_succ_fixed += (capex_charge * 1000 * capex_annual_succ(p_old, r)
                                  + inst_cap * 1000 * cfg["opex_fixed"][r])
                c_succ_var   += m.production_succ[p_old, r, y] * 1000 * succ_fuel_cost(p_old, r, y)  # LOC

        # --- GREENFIELD: CAPEX amortized over L years (vintage window),
        #     OPEX_fixed charged every year the unit is installed, fuel on prod.
        #     LOC: iterate over (route, state) pairs; capex/opex/fuel by state.
        c_green_fixed = 0.0
        c_green_var   = 0.0
        for (r, st) in m.ROUTE_STATE:
            inst_cap = green_installed_cap(r, st, y)  # for OPEX
            capex_charge = 0.0
            for sz in SIZE_MENU[r]:
                if (r, sz, st) not in m.ROUTE_SIZE_STATE:
                    continue
                capex_charge += (m.capex_vintage_green[r, sz, st, y] * sz
                                  * 1000 * capex_annual_greenfield[r])
            c_green_fixed += capex_charge + inst_cap * 1000 * cfg["opex_fixed"][r]
            c_green_var   += (m.production_greenfield[r, st, y] * 1000
                              * cost_route_by_state.get((r, y, st), 0.0))

        # --- CHARCOAL COST (3-tier supply curve, same every year)
        # cost(y) = Σ_s seg_price[s] · charcoal_delta[y, s]
        c_charcoal = sum(
            seg_price[s] * m.charcoal_delta[y, s]
            for s in m.CHARCOAL_SEG
        )

        total = (c_exist_fixed + c_exist_var
                 + c_succ_fixed + c_succ_var
                 + c_green_fixed + c_green_var
                 + c_charcoal)
        return total / discount

    m.Objective = Objective(
        expr=sum(annual_cost(y) for y in m.YEARS),
        sense=minimize,
    )

    m._ef_route       = ef_route
    m._cost_route     = cost_route_by_state   # LOC: now state-indexed
    m._biom_route     = biom_route
    m._charcoal_route = charcoal_route
    m._charcoal_seg   = (seg_width, seg_price)
    m._plant_info     = plant_info
    m._slot_info      = slot_info
    m._L              = L
    m._size_menu      = SIZE_MENU
    m._ccs_routes     = ccs_routes_in_model

    # ========================================================================
    # MODEL SIZE DIAGNOSTIC
    # ========================================================================
    n_total_vars = 0
    n_binary = 0
    n_integer = 0
    n_continuous = 0

    for v in m.component_data_objects(Var, active=True):
        n_total_vars += 1
        if v.is_binary():
            n_binary += 1
        elif v.is_integer():
            n_integer += 1
        else:
            n_continuous += 1

    n_constraints = 0
    for c in m.component_data_objects(Constraint, active=True):
        n_constraints += 1

    print("    --- Model size ---")
    print(f"    Total variables:    {n_total_vars:>8,}")
    print(f"      Binary:           {n_binary:>8,}")
    print(f"      Integer:          {n_integer:>8,}")
    print(f"      Continuous:       {n_continuous:>8,}")
    print(f"    Total constraints:  {n_constraints:>8,}")
    print("    ------------------")

    return m


def solve_model(m: ConcreteModel, log_path: str) -> dict:
    solver_name = "highs"
    solver = SolverFactory(solver_name + "nl", executable=modules.find(solver_name), solve_io="nl")

    solver_options = {
        "time_limit": 1800,      # 30 min máximo
        "mip_rel_gap": 0.02,     # aceita 2% de gap
    }

    result_solver = solver.solve(m, tee=True, options=solver_options)

    status     = str(result_solver.solver.status)
    term_cond  = str(result_solver.solver.termination_condition)
    converged  = term_cond.lower() == "optimal"
    obj_val = value(m.Objective) if converged else None

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("STEEL OPTIMIZATION MODEL — RUN LOG\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Solver status:           {status}\n")
        f.write(f"Termination condition:   {term_cond}\n")
        f.write(f"Converged to optimal:    {converged}\n")
        if obj_val is not None:
            f.write(f"Objective value (USD):   {obj_val:,.0f}\n")
        f.write("\n")
        f.write("Full solver report:\n")
        f.write("-" * 60 + "\n")
        f.write(str(result_solver))

    return {
        "status":    status,
        "term_cond": term_cond,
        "converged": converged,
        "obj_val":   obj_val,
    }

#==============================================================================
# 6. EXTRACT AND SAVE RESULTS
#==============================================================================

def extract_results(m: ConcreteModel, plants: pd.DataFrame, cfg: dict) -> dict:
    years          = cfg["YEARS"]
    routes         = cfg["routes"]
    plant_info     = m._plant_info
    slot_info      = m._slot_info
    ef_route       = m._ef_route
    charcoal_route = m._charcoal_route

    rows = []
    for p in plants["PlantID"]:
        for y in years:
            rows.append({
                "Source":        p,
                "Type":          "Existing",
                "Old_plant":     None,
                "Route":         plant_info[p]["Route"],
                "State":         plant_info[p]["UF"],   # LOC
                "Year":          y,
                "Production_kt": value(m.production_existing[p, y]),
            })
    for p_old in slot_info:
        for r in routes:
            for y in years:
                rows.append({
                    "Source":        f"{p_old}__{r}",
                    "Type":          "Successor",
                    "Old_plant":     p_old,
                    "Route":         r,
                    "State":         slot_info[p_old]["state"],   # LOC
                    "Year":          y,
                    "Production_kt": value(m.production_succ[p_old, r, y]),
                })
    for (r, st) in m.ROUTE_STATE:   # LOC: +state
        for y in years:
            rows.append({
                "Source":        f"GREENFIELD__{r}__{st}",
                "Type":          "Greenfield",
                "Old_plant":     None,
                "Route":         r,
                "State":         st,
                "Year":          y,
                "Production_kt": value(m.production_greenfield[r, st, y]),
            })
    df_prod = pd.DataFrame(rows)

    df_prod_route = (
        df_prod.groupby(["Route", "Year"])["Production_kt"].sum().unstack("Year")
    )
    df_prod_type = (
        df_prod.groupby(["Type", "Year"])["Production_kt"].sum().unstack("Year")
    )

    succ_rows = []
    for p_old, sinfo in slot_info.items():
        for y in years:
            if not (sinfo["start_year"] <= y <= sinfo["end_year"]):
                continue
            for r in routes:
                prod = value(m.production_succ[p_old, r, y])
                if prod > 1e-3:
                    succ_rows.append({
                        "Old_plant":        p_old,
                        "Old_route":        sinfo["original_route"],
                        "Old_capacity_kt":  sinfo["original_capacity"],
                        "Slot_capacity_kt": sinfo["slot_capacity"],
                        "Slot_start":       sinfo["start_year"],
                        "Slot_end":         sinfo["end_year"],
                        "Year":             y,
                        "Successor_route":  r,
                        "Production_kt":    prod,
                    })
    df_succ = pd.DataFrame(succ_rows)

    if not df_succ.empty:
        df_succ_summary = (
            df_succ.groupby(["Old_plant", "Old_route", "Old_capacity_kt",
                             "Slot_capacity_kt", "Successor_route"])
                   ["Production_kt"]
                   .sum()
                   .reset_index()
                   .rename(columns={"Production_kt": "Total_production_kt"})
                   .sort_values(["Old_plant", "Total_production_kt"], ascending=[True, False])
        )
    else:
        df_succ_summary = pd.DataFrame()

    emis_year = []
    for y in years:
        e = 0.0
        for p in plants["PlantID"]:
            e += value(m.production_existing[p, y]) * 1000 * ef_route[plant_info[p]["Route"]]
        for p_old in slot_info:
            for r in routes:
                e += value(m.production_succ[p_old, r, y]) * 1000 * ef_route[r]
        for (r, st) in m.ROUTE_STATE:   # LOC
            e += value(m.production_greenfield[r, st, y]) * 1000 * ef_route[r]
        emis_year.append({"Year": y, "Emissions_tCO2": e, "Cap_tCO2": cfg["emission_cap"][y]})
    df_emis = pd.DataFrame(emis_year)

    # ---- Charcoal demand, tier usage, marginal price and total cost per year
    seg_width, seg_price = m._charcoal_seg
    charcoal_rows = []
    for y in years:
        demand = 0.0
        for p in plants["PlantID"]:
            r = plant_info[p]["Route"]
            demand += value(m.production_existing[p, y]) * 1000 * charcoal_route[r]
        for p_old in slot_info:
            for r in routes:
                demand += value(m.production_succ[p_old, r, y]) * 1000 * charcoal_route[r]
        for (r, st) in m.ROUTE_STATE:   # LOC
            demand += value(m.production_greenfield[r, st, y]) * 1000 * charcoal_route[r]
        # Tier-by-tier usage and total cost
        tier_use = [value(m.charcoal_delta[y, s]) for s in range(len(seg_width))]
        cost_lin = sum(seg_price[s] * tier_use[s] for s in range(len(seg_width)))
        # Marginal price = price of the highest tier with any usage (or tier 1
        # if demand is zero); average price = total cost / demand.
        last_used = 0
        for s in range(len(seg_width)):
            if tier_use[s] > 1e-6:
                last_used = s
        marginal_price = seg_price[last_used]
        avg_price = (cost_lin / demand) if demand > 1e-6 else seg_price[0]
        row = {
            "Year":                       y,
            "Charcoal_demand_GJ":          demand,
            "Charcoal_demand_PJ":          demand / 1e6,
            "Marginal_price_USD_per_GJ":   marginal_price,
            "Average_price_USD_per_GJ":    avg_price,
            "Total_cost_USD":              cost_lin,
        }
        # Per-tier usage columns (in PJ, easier to read)
        for s in range(len(seg_width)):
            row[f"Tier{s+1}_used_PJ"] = tier_use[s] / 1e6
        charcoal_rows.append(row)
    df_charcoal = pd.DataFrame(charcoal_rows)

    # ---- Penetration limit report: max allowed vs actual production per group
    pen_groups = getattr(m, "_pen_groups", [])
    pen_rows = []
    for routes_in_group, ys, ye in pen_groups:
        label = "+".join(routes_in_group)
        for y in years:
            s_y = penetration_fraction(y, ys, ye)
            target = cfg["production_target"][y]
            cap = s_y * target
            actual = sum(value(m.production_succ[p_old, r, y])
                         for p_old in slot_info for r in routes_in_group)
            actual += sum(value(m.production_greenfield[r, st, y])
                          for (r, st) in m.ROUTE_STATE if r in routes_in_group)  # LOC
            pen_rows.append({
                "Group":                   label,
                "Year":                    y,
                "Window_start":            ys,
                "Window_end":              ye,
                "Sigmoid_fraction":        s_y,
                "Cap_kt":                  cap,
                "Actual_production_kt":    actual,
                "Slack_kt":                cap - actual,
                "Binding":                 (cap - actual) < 1e-3 and s_y > 1e-9,
            })
    df_penetration = pd.DataFrame(pen_rows)

    # ---- Active status: plant count and installed capacity per (slot/route/year)
    active_rows = []
    for p_old in slot_info:
        s_cap = slot_info[p_old]["slot_capacity"]
        for r in routes:
            for y in years:
                is_active = int(round(value(m.active_succ[p_old, r, y])))
                active_rows.append({
                    "Type": "Successor", "Unit": f"{p_old}__{r}",
                    "Slot": p_old, "Route": r, "Year": y,
                    "N_plants": is_active, "Installed_kt": is_active * s_cap,
                })
    for r in routes:
        for y in years:
            n_tot = sum(int(round(value(m.n_green[r, sz, st, y])))
                        for (rr, sz, st) in m.ROUTE_SIZE_STATE if rr == r)   # LOC
            cap_tot = sum(int(round(value(m.n_green[r, sz, st, y]))) * sz
                          for (rr, sz, st) in m.ROUTE_SIZE_STATE if rr == r)  # LOC
            active_rows.append({
                "Type": "Greenfield", "Unit": f"GREENFIELD__{r}",
                "Slot": None, "Route": r, "Year": y,
                "N_plants": n_tot, "Installed_kt": cap_tot,
            })
    df_active = pd.DataFrame(active_rows)

    # ---- Size breakdown: which sizes were chosen (greenfield only)   (LOC: +state)
    size_rows = []
    for (r, sz, st) in m.ROUTE_SIZE_STATE:
        for y in years:
            n_val = int(round(value(m.n_green[r, sz, st, y])))
            if n_val > 0:
                size_rows.append({
                    "Type": "Greenfield", "Slot": None, "Route": r,
                    "State": st, "Size_kt": sz, "Year": y, "N_plants": n_val,
                    "Capacity_kt": n_val * sz,
                })
    df_sizes = pd.DataFrame(size_rows)

    # LOC ---- Greenfield installed capacity by (route, state, year)
    gbs_rows = []
    for (r, sz, st) in m.ROUTE_SIZE_STATE:
        for y in years:
            n_val = int(round(value(m.n_green[r, sz, st, y])))
            if n_val > 0:
                gbs_rows.append({
                    "Route": r, "State": st, "Size_kt": sz, "Year": y,
                    "N_plants": n_val, "Installed_kt": n_val * sz,
                })
    df_greenfield_by_state = pd.DataFrame(gbs_rows)


    # ===================================================================
    # ADDITIONAL INDICATORS  (CO2 captured, fuel use, energy mix by category)
    # ===================================================================

    # ---- Fuel names for reporting (Portuguese -> English)
    FUEL_DISPLAY_NAMES = {
        "Carvao_mineral":  "Coal",
        "Carvao_vegetal":  "Charcoal",
        "Coque":           "Coke",
        "Eletricidade":    "Electricity",
        "Gas_de_coqueria": "Coke Oven Gas",
        "Gas_natural":     "Natural Gas",
        "Hidrogenio":      "Hydrogen",
        "Oleo_diesel":     "Diesel",
        "Sucata":          "Scrap",
    }

    EI         = cfg["ei"]
    ef_fuel    = cfg["ef"]
    capture_r  = cfg["CAPTURE_RATE_CCS"]

    # df_prod_route already has production by route x year (kt of steel)
    # Reuse it as the production base.

    # --- Indicator 1: CO2 captured by CCS routes (Mt/year) -----------------
    # Captured[r,y] = production_r[y] [kt] * 1000 * gross_EF[r] [tCO2/t] * capture_rate
    # where gross_EF[r] is the emission factor BEFORE capture (Σ_fuel EI[r,f]*EF[f]).
    # Loops over every route in CCS_ROUTES that is present in this config,
    # so BF-BOF-CCS and DR-NG-CCS (and any future CCS route added to
    # CCS_ROUTES) are reported individually plus as a total.
    ccs_routes_here = sorted(r for r in CCS_ROUTES if r in cfg["routes"])
    ccs_gross_ef = {
        r: sum(EI.get((r, f), 0.0) * ef_fuel.get(f, 0.0) for f in ef_fuel)
        for r in ccs_routes_here
    }
    rows_capt = []
    for y in years:
        row = {"Year": y}
        total_captured_t = 0.0
        for r in ccs_routes_here:
            prod_kt = float(df_prod_route.loc[r, y]) if r in df_prod_route.index else 0.0
            captured_t = prod_kt * 1000.0 * ccs_gross_ef[r] * capture_r
            row[f"Production_{r}_kt"]     = prod_kt
            row[f"Gross_EF_{r}_tCO2_per_t"] = ccs_gross_ef[r]
            row[f"CO2_captured_{r}_tCO2"] = captured_t
            total_captured_t += captured_t
        row["Capture_rate"]             = capture_r
        row["CO2_captured_total_tCO2"]  = total_captured_t
        row["CO2_captured_total_MtCO2"] = total_captured_t / 1e6
        rows_capt.append(row)
    df_co2_captured = pd.DataFrame(rows_capt)

    # --- Indicator 2: Total fuel use by fuel, by year (GJ) -----------------
    # For each fuel: sum over routes of EI[r,f] * production[r,y] * 1000
    fuels_in_ei = sorted({f for (r, f) in EI.keys()})
    fuel_use = {f: {} for f in fuels_in_ei}
    for y in years:
        for f in fuels_in_ei:
            total_GJ = 0.0
            for r in routes:
                prod_r_y_kt = float(df_prod_route.loc[r, y]) if r in df_prod_route.index else 0.0
                total_GJ += EI.get((r, f), 0.0) * prod_r_y_kt * 1000.0
            fuel_use[f][y] = total_GJ
    df_fuel_use = pd.DataFrame(fuel_use).T  # rows = fuels, cols = years
    df_fuel_use.index.name = "Fuel"
    # also export in PJ for readability
    df_fuel_use_PJ = df_fuel_use / 1e6
    df_fuel_use_PJ.index.name = "Fuel"
    df_fuel_use_PJ = df_fuel_use_PJ.rename(index=FUEL_DISPLAY_NAMES)

    # --- Indicator 3: Energy mix by category (Electricity / Fossil / Renewable) ---
    # Use FUEL_CATEGORY (global). Fuels not classified are dropped with a warning.
    unclassified = [f for f in fuels_in_ei if f not in FUEL_CATEGORY]
    if unclassified:
        print(f"  WARNING: fuels not in FUEL_CATEGORY (dropped from energy mix): {unclassified}")
    cat_rows = []
    for y in years:
        totals = {"Electricity": 0.0, "Fossil": 0.0, "Renewable": 0.0}
        for f in fuels_in_ei:
            cat = FUEL_CATEGORY.get(f)
            if cat is None:
                continue
            totals[cat] += fuel_use[f][y]
        grand = sum(totals.values())
        row = {"Year": y,
               "Electricity_GJ": totals["Electricity"],
               "Fossil_GJ":      totals["Fossil"],
               "Renewable_GJ":     totals["Renewable"],
               "Total_GJ":       grand}
        if grand > 0:
            row["Electricity_share_%"] = 100 * totals["Electricity"] / grand
            row["Fossil_share_%"]      = 100 * totals["Fossil"] / grand
            row["Renewable_share_%"]     = 100 * totals["Renewable"] / grand
        else:
            row["Electricity_share_%"] = row["Fossil_share_%"] = row["Renewable_share_%"] = 0.0
        row["Fossil_share_max_%"] = cfg["fossil_share_max"].get(y, float("nan")) * 100  # NEW
        cat_rows.append(row)
    df_energy_mix = pd.DataFrame(cat_rows)
    # ===================================================================

    return {
        "production_long":    df_prod,
        "production_route":   df_prod_route,
        "production_type":    df_prod_type,
        "successors":         df_succ,
        "successors_summary": df_succ_summary,
        "emissions":          df_emis,
        "charcoal_cost":      df_charcoal,
        "penetration":        df_penetration,
        "active_status":      df_active,
        "size_breakdown":     df_sizes,
        "greenfield_by_state": df_greenfield_by_state,   # LOC
        "co2_captured":       df_co2_captured,   # NEW (multi-CCS-route)
        "fuel_use_PJ":        df_fuel_use_PJ,    # NEW
        "energy_mix":         df_energy_mix,     # NEW
    }


def save_results(results: dict, output_dir: str):
    out = os.path.join(output_dir, "resultados_modelo_V20_11.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        results["production_long"].to_excel(w,    sheet_name="Production_long",    index=False)
        results["production_route"].to_excel(w,   sheet_name="Production_route")
        results["production_type"].to_excel(w,    sheet_name="Production_type")
        results["successors"].to_excel(w,         sheet_name="Successors_detail",  index=False)
        results["successors_summary"].to_excel(w, sheet_name="Successors_summary", index=False)
        results["emissions"].to_excel(w,          sheet_name="Emissions",          index=False)
        results["charcoal_cost"].to_excel(w,      sheet_name="Charcoal_cost",      index=False)
        if not results["penetration"].empty:
            results["penetration"].to_excel(w,    sheet_name="Penetration",        index=False)
        results["active_status"].to_excel(w,      sheet_name="Active_status",      index=False)
        results["size_breakdown"].to_excel(w,     sheet_name="Size_breakdown",     index=False)
        if not results.get("greenfield_by_state", pd.DataFrame()).empty:   # LOC
            results["greenfield_by_state"].to_excel(w, sheet_name="Greenfield_by_state", index=False)
        results["co2_captured"].to_excel(w,       sheet_name="CO2_captured",       index=False)
        results["fuel_use_PJ"].to_excel(w,        sheet_name="Fuel_use_PJ")
        results["energy_mix"].to_excel(w,         sheet_name="Energy_mix",         index=False)
    print(f"Results saved: {out}")
    return out


def _clean_solver_noise(df, tol: float = 1e-6):
    """Zero out values smaller than `tol` and floor anything still negative at 0."""
    df = df.where(df.abs() > tol, 0)
    df = df.clip(lower=0)
    return df


def plot_all(results: dict, cfg: dict, output_dir: str):
    # ---- Plot 1: Production by route (stacked area)
    fig, ax = plt.subplots(figsize=(12, 6))
    df = results["production_route"].T
    df = _clean_solver_noise(df)
    colors_in_order = [ROUTE_COLORS.get(r, "#999999") for r in df.columns]
    df.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0, color=colors_in_order)
    ax.set_title("Annual Production by Technology Route")
    ax.set_xlabel("Year"); ax.set_ylabel("Production (kt)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "production_by_route.png"), dpi=150)
    plt.show()

    # ---- Plot 2: Emissions trajectory vs cap
    fig, ax = plt.subplots(figsize=(10, 5))
    df_emis = results["emissions"]
    ax.plot(df_emis["Year"], df_emis["Emissions_tCO2"] / 1e6,
            "b-o", linewidth=2, label="Actual Emissions")
    ax.plot(df_emis["Year"], df_emis["Cap_tCO2"] / 1e6,
            "r--s", linewidth=2, label="Emission Cap")
    ax.set_title("Emissions Trajectory vs Cap")
    ax.set_xlabel("Year"); ax.set_ylabel("Emissions (Mt CO2)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "emissions_trajectory.png"), dpi=150)
    plt.show()

    # ---- Plot 3: Production by type
    fig, ax = plt.subplots(figsize=(12, 6))
    df_type = results["production_type"].T
    df_type = _clean_solver_noise(df_type)
    df_type.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0)
    ax.set_title("Production by Source Type")
    ax.set_xlabel("Year"); ax.set_ylabel("Production (kt)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "production_by_type.png"), dpi=150)
    plt.show()

    # ---- Plot 4: Production share by route (%)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_share = results["production_route"].T
    df_share = _clean_solver_noise(df_share)
    row_sums = df_share.sum(axis=1)
    df_share_pct = df_share.div(row_sums.where(row_sums > 1e-6, 1), axis=0) * 100
    df_share_pct.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0)
    ax.set_title("Production Share by Route (%)")
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "production_share.png"), dpi=150)
    plt.show()

    # ---- Plot 6: CO2 captured, by CCS route (stacked bars)
    df_capt = results["co2_captured"]
    ccs_cols = [c for c in df_capt.columns
                if c.startswith("CO2_captured_") and c.endswith("_tCO2")
                and c != "CO2_captured_total_tCO2"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(df_capt))
    for col in ccs_cols:
        route_name = col[len("CO2_captured_"):-len("_tCO2")]
        vals_mt = df_capt[col].to_numpy() / 1e6
        ax.bar(df_capt["Year"], vals_mt, bottom=bottom,
               color=ROUTE_COLORS.get(route_name, "#999999"), label=route_name)
        bottom += vals_mt
    ax.set_title("CO2 Captured by CCS Route")
    ax.set_xlabel("Year"); ax.set_ylabel("CO2 captured (Mt CO2/year)")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "co2_captured.png"), dpi=150)
    plt.show()

    # ---- Plot 7: Total fuel use by fuel (stacked area, PJ/year)
    df_fu = results["fuel_use_PJ"].T  # years as index, fuels as columns
    df_fu = _clean_solver_noise(df_fu)
    # Drop fuels with negligible total (e.g. Sucata if present and zero)
    df_fu = df_fu.loc[:, df_fu.sum(axis=0) > 0.01]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_in_order = [FUEL_COLORS.get(f, "#999999") for f in df_fu.columns]
    df_fu.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0, color=colors_in_order)
    ax.set_title("Total Fuel Use by Fuel (all routes combined)")
    ax.set_xlabel("Year"); ax.set_ylabel("Fuel use (PJ/year)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fuel_use_by_fuel.png"), dpi=150)
    plt.show()

    # ---- Plot 8: Energy mix by category (% share, stacked area)
    df_mix_full = results["energy_mix"].set_index("Year")
    df_mix = df_mix_full[["Electricity_share_%", "Fossil_share_%", "Renewable_share_%"]]
    df_mix.columns = ["Electricity", "Fossil", "Renewable"]
    fig, ax = plt.subplots(figsize=(12, 6))
    df_mix.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0,
                color=["#F2B705", "#5A5A5A", "#2E7D32"])
    if "Fossil_share_max_%" in df_mix_full.columns:  # NEW — show the cap
        ax.plot(df_mix_full.index, df_mix_full["Fossil_share_max_%"],
                "r--", linewidth=2, label="Fossil share cap")
    ax.set_title("Energy Mix by Category (% of total energy use)")
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "energy_mix.png"), dpi=150)
    plt.show()

    # ---- Plot 5: Charcoal — demand vs supply curve and marginal price
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    df_ch = results["charcoal_cost"]

    # left: tier usage stacked over time (PJ/year)
    tier_cols = [c for c in df_ch.columns if c.startswith("Tier") and c.endswith("_PJ")]
    df_tiers = df_ch.set_index("Year")[tier_cols]
    df_tiers = df_tiers.clip(lower=0)   # LOC-fix: solver returns tiny negatives (~-1e-9) that break stacked area
    df_tiers.plot(kind="area", stacked=True, ax=ax1, alpha=0.8)
    ax1.set_title("Charcoal demand by tier")
    ax1.set_xlabel("Year"); ax1.set_ylabel("Demand (PJ/year)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    # right: marginal price and average price over time
    ax2.plot(df_ch["Year"], df_ch["Marginal_price_USD_per_GJ"],
             "g-o", linewidth=2, label="Marginal price (highest tier used)")
    ax2.plot(df_ch["Year"], df_ch["Average_price_USD_per_GJ"],
             "b--s", linewidth=2, label="Average price (total cost / demand)")
    for tp in CHARCOAL_TIER_PRICE:
        ax2.axhline(tp, color="gray", linestyle=":", alpha=0.5)
    ax2.set_title("Charcoal price over time")
    ax2.set_xlabel("Year"); ax2.set_ylabel("Price (USD/GJ)")
    ax2.legend(); ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "charcoal_supply.png"), dpi=150)
    plt.show()

    # ---- Plot 6: Penetration limit — cap (sigmoid) vs actual production
    df_pen = results["penetration"]
    if not df_pen.empty:
        groups = df_pen["Group"].unique()
        fig, ax = plt.subplots(figsize=(11, 5))
        for g in groups:
            sub = df_pen[df_pen["Group"] == g].sort_values("Year")
            line, = ax.plot(sub["Year"], sub["Cap_kt"] / 1000,
                            "--", linewidth=2, label=f"{g}: cap")
            ax.plot(sub["Year"], sub["Actual_production_kt"] / 1000,
                    "-o", linewidth=2, color=line.get_color(),
                    label=f"{g}: actual")
        ax.set_title("Technology penetration: cap (sigmoid) vs actual production")
        ax.set_xlabel("Year"); ax.set_ylabel("Production (Mt/year)")
        ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "penetration.png"), dpi=150)
        plt.show()

    # ---- Plot: Greenfield installed capacity by STATE   (LOC, from V20_14)
    df_gs = results.get("greenfield_by_state", pd.DataFrame())
    if not df_gs.empty:
        df_state_year = (
            df_gs.groupby(["State", "Year"])["Installed_kt"].max()
                 .unstack("Year").fillna(0.0)
        )
        fig, ax = plt.subplots(figsize=(12, 6))
        df_state_year.T.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0)
        ax.set_title("Greenfield Installed Capacity by State")
        ax.set_xlabel("Year"); ax.set_ylabel("Installed capacity (kt)")
        ax.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "greenfield_by_state.png"), dpi=150)
        plt.show()


# ============================================================================
# 7. REFERENCE SCENARIO (REF / BAU) — no optimisation, frozen 2023 route mix
# ============================================================================
def run_reference_scenario(cfg: dict, output_dir: str):
    """
    Build a Reference scenario assuming the SAME route distribution as
    the base year throughout the horizon. Production grows with demand;
    route shares stay constant. No optimisation — just proportional scaling.
    """
    years         = cfg["YEARS"]
    routes        = cfg["routes"]
    EI            = cfg["ei"]
    ef_fuel       = cfg["ef"]
    capture_r     = cfg["CAPTURE_RATE_CCS"]
    steel_demand  = cfg["production_target"]
    emission_cap  = cfg["emission_cap"]
    plants        = cfg["plants"]
    base_year     = cfg["BASE_YEAR"]

    # --- (1) Compute 2023 route distribution from the optimised MIT scenario --
    # The REF scenario uses the SAME 2023 starting point as the MIT scenarios
    # (so that diverging emissions trajectories reflect the policy effect, not
    # a different base-year allocation). The shares are read from the
    # Production_route sheet of an MIT scenario output file.
    MIT_BASELINE_FILE = os.path.join(output_dir, "resultados_modelo_V20_11.xlsx")
    if not os.path.exists(MIT_BASELINE_FILE):
        raise RuntimeError(
            f"Reference scenario: cannot find {MIT_BASELINE_FILE}. "
            f"Run the optimised MIT scenario first."
        )
    df_mit_prod = pd.read_excel(MIT_BASELINE_FILE, "Production_route")
    # Filter out a possible "Total" row, keep only real routes
    df_mit_prod = df_mit_prod[df_mit_prod["Route"] != "Total"].copy()
    df_mit_prod = df_mit_prod.set_index("Route")
    base_year_col = base_year  # int header from the Excel sheet

    prod_base = {r: 0.0 for r in routes}
    for r in routes:
        if r in df_mit_prod.index:
            prod_base[r] = float(df_mit_prod.loc[r, base_year_col])
        else:
            prod_base[r] = 0.0

    total_base = sum(prod_base.values())
    if total_base <= 0:
        raise RuntimeError(
            "Reference scenario: base-year production is zero. "
            "Check the MIT baseline file."
        )
    route_share = {r: prod_base[r] / total_base for r in routes}

    print(f"\n[REF scenario] Base-year ({base_year}) route shares:")
    for r in routes:
        if route_share[r] > 1e-6:
            print(f"    {r:<14} {route_share[r]*100:>5.1f}%")

    # --- (2) Apply constant share to total demand each year ----------------
    # production_REF[r, y] = route_share[r] * steel_demand[y]
    rows_prod = []
    for y in years:
        demand_y = float(steel_demand[y])
        for r in routes:
            rows_prod.append({"Route": r, "Year": y,
                              "Production_kt": route_share[r] * demand_y})
    df_prod_long = pd.DataFrame(rows_prod)
    df_prod_route = df_prod_long.pivot(index="Route", columns="Year",
                                       values="Production_kt").reindex(routes)

    # --- (3) Emissions per route per year ---------------------------------
    # ef_route[r] = sum_f EI[r,f] * ef_fuel[f]
    # For any route in CCS_ROUTES, apply capture (1 - rate). In REF, CCS
    # share is typically zero anyway, but this is kept for completeness in
    # case the base year already has some CCS production.
    def ef_route(r):
        e = sum(EI.get((r, f), 0.0) * ef_fuel.get(f, 0.0) for f in ef_fuel)
        if r in CCS_ROUTES:
            e *= (1 - capture_r)
        return e

    rows_emis = []
    for y in years:
        total_tCO2 = sum(
            float(df_prod_route.loc[r, y]) * 1000.0 * ef_route(r)
            for r in routes
        )
        rows_emis.append({"Year": y,
                          "Emissions_tCO2": total_tCO2,
                          "Cap_tCO2":       float(emission_cap[y])})
    df_emis = pd.DataFrame(rows_emis)

    # --- (4) Fuel use by fuel by year (GJ, then PJ) -----------------------
    # Translation map to keep fuel names consistent with the optimised
    # scenario output (English labels).
    FUEL_DISPLAY_NAMES = {
        "Carvao_mineral":  "Coal",
        "Carvao_vegetal":  "Charcoal",
        "Coque":           "Coke",
        "Eletricidade":    "Electricity",
        "Gas_de_coqueria": "Coke Oven Gas",
        "Gas_natural":     "Natural Gas",
        "Hidrogenio":      "Hydrogen",
        "Oleo_diesel":     "Diesel",
        "Sucata":          "Scrap",
    }

    fuels = sorted({f for (_, f) in EI.keys()})
    fuel_use = {f: {} for f in fuels}
    for y in years:
        for f in fuels:
            total_GJ = sum(
                EI.get((r, f), 0.0) * float(df_prod_route.loc[r, y]) * 1000.0
                for r in routes
            )
            fuel_use[f][y] = total_GJ
    df_fuel_use_PJ = pd.DataFrame(fuel_use).T / 1e6
    df_fuel_use_PJ.index.name = "Fuel"
    df_fuel_use_PJ = df_fuel_use_PJ.rename(index=FUEL_DISPLAY_NAMES)

    # --- (5) Energy mix by category ---------------------------------------
    cat_rows = []
    for y in years:
        totals = {"Electricity": 0.0, "Fossil": 0.0, "Renewable": 0.0}
        for f in fuels:
            cat = FUEL_CATEGORY.get(f)
            if cat is None:
                continue
            totals[cat] += fuel_use[f][y]
        grand = sum(totals.values())
        row = {"Year": y,
               "Electricity_GJ": totals["Electricity"],
               "Fossil_GJ":      totals["Fossil"],
               "Renewable_GJ":   totals["Renewable"],
               "Total_GJ":       grand}
        if grand > 0:
            row["Electricity_share_%"] = 100 * totals["Electricity"] / grand
            row["Fossil_share_%"]      = 100 * totals["Fossil"] / grand
            row["Renewable_share_%"]   = 100 * totals["Renewable"] / grand
        else:
            row["Electricity_share_%"] = row["Fossil_share_%"] = row["Renewable_share_%"] = 0.0
        row["Fossil_share_max_%"] = cfg["fossil_share_max"].get(y, float("nan")) * 100  # NEW
        cat_rows.append(row)
    df_energy_mix = pd.DataFrame(cat_rows)

    # --- (6) CO2 captured (typically zero in REF, kept for indicator parity,
    #         reported per CCS route, same structure as the MIT scenario) ---
    ccs_routes_here = sorted(r for r in CCS_ROUTES if r in routes)
    ccs_gross_ef = {
        r: sum(EI.get((r, f), 0.0) * ef_fuel.get(f, 0.0) for f in ef_fuel)
        for r in ccs_routes_here
    }
    rows_capt = []
    for y in years:
        row = {"Year": y}
        total_captured_t = 0.0
        for r in ccs_routes_here:
            prod_kt = float(df_prod_route.loc[r, y]) if r in df_prod_route.index else 0.0
            captured_t = prod_kt * 1000.0 * ccs_gross_ef[r] * capture_r
            row[f"Production_{r}_kt"]     = prod_kt
            row[f"CO2_captured_{r}_tCO2"] = captured_t
            total_captured_t += captured_t
        row["CO2_captured_total_tCO2"]  = total_captured_t
        row["CO2_captured_total_MtCO2"] = total_captured_t / 1e6
        rows_capt.append(row)
    df_co2_captured = pd.DataFrame(rows_capt)

    # --- (7) Route shares table (constant by definition) -------------------
    df_shares = pd.DataFrame([
        {"Route": r, "Share_%": route_share[r] * 100} for r in routes
    ])

    # --- (8) Save Excel ----------------------------------------------------
    out_xlsx = os.path.join(output_dir, "resultados_REFERENCE.xlsx")
    with pd.ExcelWriter(out_xlsx) as w:
        df_prod_long.to_excel(w,    sheet_name="Production_long",     index=False)
        df_prod_route.to_excel(w,   sheet_name="Production_route")
        df_shares.to_excel(w,       sheet_name="Route_shares_base",   index=False)
        df_emis.to_excel(w,         sheet_name="Emissions",           index=False)
        df_fuel_use_PJ.to_excel(w,  sheet_name="Fuel_use_PJ")
        df_energy_mix.to_excel(w,   sheet_name="Energy_mix",          index=False)
        df_co2_captured.to_excel(w, sheet_name="CO2_captured",        index=False)
    print(f"[REF scenario] Results saved: {out_xlsx}")

    # --- (9) Plots --------------------------------------------------------
    # Production by route (stacked area)
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = df_prod_route.T
    colors_in_order = [ROUTE_COLORS.get(r, "#999999") for r in df_plot.columns]
    df_plot.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0, color=colors_in_order)
    ax.set_title("REF scenario — Annual Production by Technology Route")
    ax.set_xlabel("Year"); ax.set_ylabel("Production (kt)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "REF_production_by_route.png"), dpi=150)
    plt.close(fig)

    # Emissions vs cap
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_emis["Year"], df_emis["Emissions_tCO2"]/1e6, "b-o", linewidth=2,
            label="REF emissions")
    ax.plot(df_emis["Year"], df_emis["Cap_tCO2"]/1e6, "r--s", linewidth=2,
            label="Emission cap (reference only)")
    ax.set_title("REF scenario — Emissions Trajectory vs Cap")
    ax.set_xlabel("Year"); ax.set_ylabel("Emissions (Mt CO2)")
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "REF_emissions_trajectory.png"), dpi=150)
    plt.close(fig)

    # Fuel use stacked
    df_fu = df_fuel_use_PJ.T
    df_fu = df_fu.loc[:, df_fu.sum(axis=0) > 0.01]
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_in_order = [FUEL_COLORS.get(f, "#999999") for f in df_fu.columns]
    df_fu.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0, color=colors_in_order)
    ax.set_title("REF scenario — Total Fuel Use by Fuel")
    ax.set_xlabel("Year"); ax.set_ylabel("Fuel use (PJ/year)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "REF_fuel_use_by_fuel.png"), dpi=150)
    plt.close(fig)

    # Energy mix
    df_mix_full = df_energy_mix.set_index("Year")
    df_mix = df_mix_full[["Electricity_share_%", "Fossil_share_%", "Renewable_share_%"]]
    df_mix.columns = ["Electricity", "Fossil", "Renewable"]
    fig, ax = plt.subplots(figsize=(12, 6))
    df_mix.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0,
                color=["#F2B705", "#5A5A5A", "#2E7D32"])
    if "Fossil_share_max_%" in df_mix_full.columns:  # NEW — show the cap
        ax.plot(df_mix_full.index, df_mix_full["Fossil_share_max_%"],
                "r--", linewidth=2, label="Fossil share cap")
    ax.set_title("REF scenario — Energy Mix by Category")
    ax.set_xlabel("Year"); ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "REF_energy_mix.png"), dpi=150)
    plt.close(fig)

    print(f"[REF scenario] Plots saved to: {output_dir}")
    return {
        "production_route": df_prod_route,
        "emissions":        df_emis,
        "fuel_use_PJ":      df_fuel_use_PJ,
        "energy_mix":       df_energy_mix,
        "co2_captured":     df_co2_captured,
        "route_shares":     df_shares,
    }


#==============================================================================
# 8. MAIN
#==============================================================================

def main(config_file: str = None, output_dir: str = None):
    """Run one full scenario: load config+plants, build, solve, extract,
    save results and plots, then run the REF scenario — all into
    `output_dir`. Defaults to the module-level CONFIG_FILE/OUTPUT_DIR for
    backward compatibility (single-scenario runs)."""
    config_file = config_file or CONFIG_FILE
    output_dir  = output_dir or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*78}\n>>> SCENARIO: {os.path.basename(config_file)}\n>>> Output:   {output_dir}\n{'='*78}")

    print(">>> Loading config...")
    cfg = load_config(config_file)
    print(f"    Years: {cfg['YEAR_START']}–{cfg['YEAR_END']}")
    print(f"    Routes: {cfg['routes']}")
    print("    CAPEX by route (USD/t) — retrofit vs greenfield:")
    for r in cfg["routes"]:
        cr = cfg["capex_retrofit"][r]
        cg = cfg["capex_greenfield"][r]
        print(f"      {r:<14} retrofit={cr:>6.0f}   greenfield={cg:>6.0f}")
    print(f"    Max ramp-down: {cfg['MAX_RAMP_DOWN']*100:.0f}%/year (skipped in last active year)")
    print(f"    Charcoal supply: 3 tiers, "
          f"{CHARCOAL_TIER_WIDTH_PJ} PJ at {CHARCOAL_TIER_PRICE} USD/GJ "
          f"({CHARCOAL_ABOVE_CAP} above the top tier)")
    ccs_in_routes = [r for r in CCS_ROUTES if r in cfg["routes"]]
    print(f"    CCS routes: {ccs_in_routes} "
          f"(capture rate: {cfg['CAPTURE_RATE_CCS']*100:.0f}%)")
    if PENETRATION_LIMITS:
        print(f"    Penetration limits (smoothstep sigmoid):")
        for k, v in PENETRATION_LIMITS.items():
            print(f"      {k}: 0 in {v['start']} -> 1 in {v['end']}")
    fsm = cfg["fossil_share_max"]
    print(f"    Fossil share cap: {fsm[cfg['YEAR_START']]*100:.0f}% in {cfg['YEAR_START']} "
          f"-> {fsm[cfg['YEAR_END']]*100:.0f}% in {cfg['YEAR_END']}")

    print(">>> Loading plants...")
    plants = load_plants(PLANTS_FILE)   # same fleet for every scenario — only the config varies
    print(f"    {len(plants)} plants loaded.")

    print(">>> Building model...")
    m = build_model(plants, cfg)

    print(">>> Solving...")
    log_path = os.path.join(output_dir, "optimization_log.txt")
    info = solve_model(m, log_path)

    if not info["converged"]:
        print(f"!!! Did NOT converge: {info['term_cond']}")
        print(f"    See log: {log_path}")
        return False

    print(f">>> Optimal cost (NPV USD): {info['obj_val']:,.0f}")
    print(">>> Extracting results...")
    results = extract_results(m, plants, cfg)
    save_results(results, output_dir)
    plot_all(results, cfg, output_dir)
    plot_production_by_state(results, output_dir)   # LOC

    print(">>> Running REFERENCE scenario...")
    cfg_ref = load_config(config_file)
    cfg_ref["plants"] = plants
    run_reference_scenario(cfg_ref, output_dir)

    print(f">>> Done: {os.path.basename(config_file)}")
    return True


def scenario_label(path: str) -> str:
    """Derive a short, filesystem-safe folder name from a config filename.
    Strips the common 'Input_Model_Config_12_horizonte_2070' prefix (and
    any leading numeric upload-id prefix) and keeps whatever scenario tag
    remains, e.g. '..._MIT_V2_-H2_27_GN_15_EL_27.xlsx' -> 'MIT_V2_-H2_27_GN_15_EL_27'.
    Falls back to the full stem if nothing recognizable is found.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    stem = re.sub(r"^\d+_", "", stem)  # drop leading upload-id timestamp, if any
    marker = "Input_Model_Config_12_horizonte_2070"
    if marker in stem:
        tag = stem.split(marker, 1)[1].lstrip("_")
        return tag if tag else "base"
    return stem


def plot_production_by_state(results: dict = None, output_dir: str = None,
                            min_kt: float = 1.0, save_table: bool = True):
    """LOC: stacked-area plot of total steel production by UF (existing +
    successor + greenfield). Prefers in-memory `results`; falls back to the
    Production_long sheet of a saved workbook in output_dir if given.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt

    if output_dir is None:
        output_dir = OUTPUT_DIR

    if results is not None:
        df = results["production_long"].copy()
    else:
        print("[plot_production_by_state] No results passed — call "
              "plot_production_by_state(results=results) after a run.")
        return None

    if "State" not in df.columns:
        print("[plot_production_by_state] Production_long has no 'State' column.")
        return None

    df = df[df["Production_kt"] > 1e-6]
    if df.empty:
        print("[plot_production_by_state] No production found — nothing to plot.")
        return None

    df_state_year = (df.groupby(["State", "Year"])["Production_kt"].sum()
                       .unstack("Year").fillna(0.0))

    peak = df_state_year.max(axis=1)
    keep = peak[peak >= min_kt].sort_values(ascending=False).index
    if len(keep) == 0:
        print(f"[plot_production_by_state] No state reaches {min_kt} kt.")
        return None
    df_state_year = df_state_year.loc[keep]

    fig, ax = plt.subplots(figsize=(12, 6))
    df_state_year.T.plot(kind="area", stacked=True, ax=ax, alpha=0.8, linewidth=0)
    ax.set_title("Steel Production by State — full fleet "
                 "(existing, successor and greenfield)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Production (kt/year)")
    ax.set_xlim(df_state_year.columns.min(), df_state_year.columns.max())
    ax.set_ylim(0, None)
    ax.legend(title="State", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "production_by_state.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plot_production_by_state] Saved: {path}")

    if save_table:
        out = os.path.join(output_dir, "production_by_state.xlsx")
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            df_state_year.to_excel(w, sheet_name="Production_by_state")
        print(f"[plot_production_by_state] Saved: {out}")

    return df_state_year


if __name__ == "__main__":
    # ==========================================================================
    # BATCH MODE: run every *.xlsx config file found in INPUT_DIR (falls back
    # to the single CONFIG_FILE above if INPUT_DIR doesn't exist / is empty).
    # Each config gets its own output subfolder under OUTPUT_DIR, named after
    # its scenario tag (see scenario_label). Uses the SAME plants file
    # (PLANTS_FILE) for every run — only the config workbook varies.
    # ==========================================================================
    import glob
    import re

    INPUT_DIR = os.path.join(BASE_DIR, "input", "batch")  # >>> put all scenario .xlsx files here <<<
    config_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.xlsx")))

    if not config_files:
        print(f">>> No .xlsx files found in {INPUT_DIR} — running single CONFIG_FILE instead.")
        main()
    else:
        print(f">>> BATCH MODE: {len(config_files)} scenario file(s) found in {INPUT_DIR}")
        summary = []
        for cf in config_files:
            label = scenario_label(cf)
            out_dir = os.path.join(OUTPUT_DIR, label)
            try:
                ok = main(config_file=cf, output_dir=out_dir)
                summary.append((label, "OK" if ok else "DID NOT CONVERGE"))
            except Exception as e:
                print(f"!!! ERROR running {label}: {e}")
                summary.append((label, f"ERROR: {e}"))

        print(f"\n{'='*78}\n>>> BATCH SUMMARY\n{'='*78}")
        for label, status in summary:
            print(f"  {label:<40} {status}")