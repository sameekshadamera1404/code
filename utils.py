# ✅ Dynamic normalization
def normalize(value, min_val, max_val):
    if value is None:
        return 0.0

    value = float(value)

    if max_val == min_val:
        return 0.0

    norm = (value - min_val) / (max_val - min_val)
    return max(0.0, min(norm, 1.0))


# ✅ Risk computation
def compute_risk(likelihood, impact_rating, severity, stats):
    raw_probability = normalize(
        likelihood,
        stats["likelihood_min"],
        stats["likelihood_max"]
    )
    probability = 1 + 4 * raw_probability

    raw_impact = normalize(
        impact_rating,
        stats["impact_min"],
        stats["impact_max"]
    )
    impact = raw_impact * 5

    severity_norm = normalize(
        severity,
        stats["severity_min"],
        stats["severity_max"]
    )

    risk_score = probability * impact * severity_norm

    return probability, impact, risk_score


# ✅ Controls aggregation (average instead of max)
def aggregate_controls(df, stats):
    results = {}

    for _, row in df.iterrows():
        prob, impact, risk = compute_risk(
            row["likelihood"],
            row["impact_rating"],
            row["severity"],
            stats
        )

        control = row["control_name"]

        if control not in results:
            results[control] = {
                "probability": prob,
                "impact": impact,
                "risk_score": risk,
                "count": 1
            }
        else:
            results[control]["probability"] += prob
            results[control]["impact"] += impact
            results[control]["risk_score"] += risk
            results[control]["count"] += 1

    # ✅ finalize averages
    for c in results:
        results[c]["probability"] /= results[c]["count"]
        results[c]["impact"] /= results[c]["count"]
        results[c]["risk_score"] /= results[c]["count"]

    return results


# ✅ Threats aggregation (average instead of max)
def aggregate_threats(df, stats):
    results = {}

    for _, row in df.iterrows():
        prob, impact, risk = compute_risk(
            row["likelihood"],
            row["impact_rating"],
            row["severity"],
            stats
        )

        threat = row["threat_name"]

        if threat not in results:
            results[threat] = {
                "probability": prob,
                "impact": impact,
                "risk_score": risk,
                "count": 1
            }
        else:
            results[threat]["probability"] += prob
            results[threat]["impact"] += impact
            results[threat]["risk_score"] += risk
            results[threat]["count"] += 1

    # ✅ finalize averages
    for t in results:
        results[t]["probability"] /= results[t]["count"]
        results[t]["impact"] /= results[t]["count"]
        results[t]["risk_score"] /= results[t]["count"]

    return results