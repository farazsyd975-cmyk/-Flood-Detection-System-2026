/**
 * ==============================================================================
 * FLOOD DETECTION SYSTEM - MAIN FRONTEND CONTROLLER
 * Coordinates UI inputs, live weather API, simulation modes, and ML inference.
 * ==============================================================================
 */

document.addEventListener("DOMContentLoaded", function () {
    // DOM Element References
    const locationSelect = document.getElementById("location-select");
    const stationBadge = document.getElementById("station-badge");
    const geoRiver = document.getElementById("geo-river");
    const geoCoords = document.getElementById("geo-coords");
    const geoDangerStage = document.getElementById("geo-danger-stage");
    const globalDataSource = document.getElementById("global-data-source");

    const btnFetchLive = document.getElementById("btn-fetch-live");
    const toggleSimBtn = document.getElementById("toggle-sim-mode");
    const simPresetsPanel = document.getElementById("sim-presets-panel");
    const predictionForm = document.getElementById("prediction-form");
    const btnPredict = document.getElementById("btn-predict");

    // Environmental Parameter Inputs & Badges
    const inputs = {
        rainfall: document.getElementById("input-rainfall"),
        riverLevel: document.getElementById("input-river-level"),
        soilMoisture: document.getElementById("input-soil-moisture"),
        prevRain: document.getElementById("input-prev-rain"),
        temp: document.getElementById("input-temp"),
        humidity: document.getElementById("input-humidity"),
        elevation: document.getElementById("input-elevation"),
        distance: document.getElementById("input-distance"),
        dataSource: document.getElementById("input-data-source")
    };

    const badges = {
        rainfall: document.getElementById("val-rainfall"),
        riverLevel: document.getElementById("val-river-level"),
        soilMoisture: document.getElementById("val-soil-moisture"),
        prevRain: document.getElementById("val-prev-rain"),
        temp: document.getElementById("val-temp"),
        humidity: document.getElementById("val-humidity")
    };

    // Result DOM Elements
    const resultBox = document.querySelector(".result-box");
    const resultRiskLabel = document.getElementById("result-risk-label");
    const resultConfidenceBadge = document.getElementById("result-confidence-badge");
    const resultSummary = document.getElementById("result-summary");
    const resultAdvisory = document.getElementById("result-advisory");
    const resultAction = document.getElementById("result-action");
    const resultTimestamp = document.getElementById("result-timestamp");

    // Probability Bars
    const probBars = {
        Low: document.getElementById("bar-prob-low"),
        Moderate: document.getElementById("bar-prob-mod"),
        High: document.getElementById("bar-prob-high"),
        Severe: document.getElementById("bar-prob-sev")
    };

    const probTexts = {
        Low: document.getElementById("txt-prob-low"),
        Moderate: document.getElementById("txt-prob-mod"),
        High: document.getElementById("txt-prob-high"),
        Severe: document.getElementById("txt-prob-sev")
    };

    let isSimulationMode = false;

    // -----------------------------------------------------------------------
    // 1. Station Selection Handlers
    // -----------------------------------------------------------------------

    function selectStation(cityName, refreshWeather = true) {
        if (!cityName) return;

        // Update dropdown
        if (locationSelect && locationSelect.value !== cityName) {
            locationSelect.value = cityName;
        }

        // Update Quick-Select Pills
        document.querySelectorAll(".station-pill").forEach(pill => {
            if (pill.getAttribute("data-city") === cityName) {
                pill.classList.add("active");
            } else {
                pill.classList.remove("active");
            }
        });

        // Focus map
        if (typeof centerMapOnStation === "function") {
            centerMapOnStation(cityName);
        }

        if (refreshWeather) {
            loadStationTelemetry(cityName);
        }
    }

    // Dropdown change event
    if (locationSelect) {
        locationSelect.addEventListener("change", function () {
            selectStation(this.value, true);
        });
    }

    // Quick Select Pills click event
    document.querySelectorAll(".station-pill").forEach(pill => {
        pill.addEventListener("click", function () {
            const city = this.getAttribute("data-city");
            selectStation(city, true);
        });
    });

    // Expose global hook for Leaflet map popup clicks
    window.handleMarkerSelected = function (cityName) {
        selectStation(cityName, true);
    };

    // -----------------------------------------------------------------------
    // 2. Fetch Environmental Telemetry (Live / Baseline)
    // -----------------------------------------------------------------------

    function loadStationTelemetry(cityName) {
        if (!cityName) return;

        if (btnFetchLive) {
            btnFetchLive.disabled = true;
            btnFetchLive.textContent = "⏳ Fetching...";
        }

        fetch(`/api/weather/${encodeURIComponent(cityName)}`)
            .then(res => res.json())
            .then(payload => {
                if (payload.success && payload.data) {
                    const d = payload.data;

                    // Update Callout Information
                    if (stationBadge) stationBadge.textContent = `${cityName} (${d.elevation_m}m MSL)`;
                    if (geoRiver) geoRiver.textContent = d.nearest_river;
                    if (geoCoords) geoCoords.textContent = `${d.latitude}°N, ${d.longitude}°E`;
                    if (geoDangerStage) geoDangerStage.textContent = `${d.danger_river_level_m} m`;
                    if (globalDataSource) globalDataSource.textContent = d.data_source;
                    if (inputs.dataSource) inputs.dataSource.value = d.data_source;

                    // Update Topographical hidden inputs
                    if (inputs.elevation) inputs.elevation.value = d.elevation_m;
                    if (inputs.distance) inputs.distance.value = d.distance_to_river_km;

                    // Populate Input Sliders & Badges
                    updateInput("rainfall", d.rainfall_mm, "mm");
                    updateInput("riverLevel", d.river_water_level_m, "m");
                    updateInput("soilMoisture", d.soil_moisture_pct, "%");
                    updateInput("prevRain", d.prev_rainfall_7d_mm, "mm");
                    updateInput("temp", d.temp_c, "°C");
                    updateInput("humidity", d.humidity_pct, "%");

                    // Trigger automatic prediction for refreshed data
                    executePrediction(false);
                }
            })
            .catch(err => {
                console.error("[Telemetry Error] Failed to fetch environmental telemetry:", err);
            })
            .finally(() => {
                if (btnFetchLive) {
                    btnFetchLive.disabled = false;
                    btnFetchLive.textContent = "🔄 Refresh Live Data";
                }
            });
    }

    if (btnFetchLive) {
        btnFetchLive.addEventListener("click", function () {
            const currentCity = locationSelect ? locationSelect.value : "Jalandhar";
            loadStationTelemetry(currentCity);
        });
    }

    // -----------------------------------------------------------------------
    // 3. Slider Binding & Value Display Updates
    // -----------------------------------------------------------------------

    function updateInput(key, val, unit) {
        if (inputs[key]) inputs[key].value = val;
        if (badges[key]) badges[key].textContent = `${val} ${unit}`;
    }

    function bindSlider(inputElem, badgeElem, unit) {
        if (!inputElem || !badgeElem) return;
        inputElem.addEventListener("input", function () {
            badgeElem.textContent = `${this.value} ${unit}`;
            if (inputs.dataSource) {
                inputs.dataSource.value = "Manual Parameter Simulation";
                if (globalDataSource) globalDataSource.textContent = "Manual Parameter Simulation";
            }
        });
    }

    bindSlider(inputs.rainfall, badges.rainfall, "mm");
    bindSlider(inputs.riverLevel, badges.riverLevel, "m");
    bindSlider(inputs.soilMoisture, badges.soilMoisture, "%");
    bindSlider(inputs.prevRain, badges.prevRain, "mm");
    bindSlider(inputs.temp, badges.temp, "°C");
    bindSlider(inputs.humidity, badges.humidity, "%");

    // -----------------------------------------------------------------------
    // 4. Simulation Mode & Scenario Presets (Viva Demonstration Feature)
    // -----------------------------------------------------------------------

    if (toggleSimBtn) {
        toggleSimBtn.addEventListener("click", function () {
            isSimulationMode = !isSimulationMode;

            // Enable or disable input sliders
            Object.values(inputs).forEach(inp => {
                if (inp && inp.type === "range") {
                    inp.disabled = !isSimulationMode;
                }
            });

            if (isSimulationMode) {
                this.classList.add("active");
                this.textContent = "🔒 Lock Parameters (Live Mode)";
                if (simPresetsPanel) simPresetsPanel.style.display = "block";
            } else {
                this.classList.remove("active");
                this.textContent = "✏️ Tweak Parameters (Simulation Mode)";
                if (simPresetsPanel) simPresetsPanel.style.display = "none";
            }
        });
    }

    // Viva Quick Presets
    document.querySelectorAll(".btn-preset").forEach(btn => {
        btn.addEventListener("click", function () {
            const scenario = this.getAttribute("data-scenario");
            applyScenarioPreset(scenario);
        });
    });

    function applyScenarioPreset(scenario) {
        if (inputs.dataSource) {
            inputs.dataSource.value = `Simulation: ${scenario.toUpperCase()}`;
            if (globalDataSource) globalDataSource.textContent = `Simulation (${scenario})`;
        }

        switch (scenario) {
            case "dry":
                updateInput("rainfall", 2.0, "mm");
                updateInput("riverLevel", 1.8, "m");
                updateInput("soilMoisture", 28.0, "%");
                updateInput("prevRain", 5.0, "mm");
                updateInput("temp", 33.0, "°C");
                updateInput("humidity", 45.0, "%");
                break;
            case "monsoon":
                updateInput("rainfall", 45.0, "mm");
                updateInput("riverLevel", 2.8, "m");
                updateInput("soilMoisture", 62.0, "%");
                updateInput("prevRain", 60.0, "mm");
                updateInput("temp", 28.5, "°C");
                updateInput("humidity", 78.0, "%");
                break;
            case "heavy":
                updateInput("rainfall", 95.0, "mm");
                updateInput("riverLevel", 4.2, "m");
                updateInput("soilMoisture", 82.0, "%");
                updateInput("prevRain", 130.0, "mm");
                updateInput("temp", 26.0, "°C");
                updateInput("humidity", 90.0, "%");
                break;
            case "historic": // July 2023 Sutlej/Beas floods
                updateInput("rainfall", 185.0, "mm");
                updateInput("riverLevel", 5.8, "m");
                updateInput("soilMoisture", 96.0, "%");
                updateInput("prevRain", 240.0, "mm");
                updateInput("temp", 24.5, "°C");
                updateInput("humidity", 96.0, "%");
                break;
        }

        executePrediction(true);
    }

    // -----------------------------------------------------------------------
    // 5. ML Inference Execution
    // -----------------------------------------------------------------------

    function executePrediction(scrollToResult = false) {
        const cityName = locationSelect ? locationSelect.value : "Jalandhar";

        const payload = {
            location_name: cityName,
            rainfall_mm: parseFloat(inputs.rainfall.value),
            river_water_level_m: parseFloat(inputs.riverLevel.value),
            soil_moisture_pct: parseFloat(inputs.soilMoisture.value),
            prev_rainfall_7d_mm: parseFloat(inputs.prevRain.value),
            temp_c: parseFloat(inputs.temp.value),
            humidity_pct: parseFloat(inputs.humidity.value),
            elevation_m: parseFloat(inputs.elevation.value),
            distance_to_river_km: parseFloat(inputs.distance.value),
            data_source: inputs.dataSource.value || "Manual Input"
        };

        if (btnPredict) {
            btnPredict.disabled = true;
            btnPredict.textContent = "⚡ Running Random Forest Inference...";
        }

        fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        })
            .then(res => res.json())
            .then(data => {
                if (data.success && data.result) {
                    renderPredictionResult(data.result);

                    // Update Leaflet map marker color & telemetry dynamically
                    if (typeof updateMarkerRisk === "function") {
                        updateMarkerRisk(
                            cityName,
                            data.result.prediction.risk_level,
                            data.result.prediction.risk_label,
                            data.result.prediction.confidence_pct,
                            data.result.inputs
                        );
                    }

                    if (scrollToResult && resultBox) {
                        resultBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
                    }
                } else {
                    alert("Prediction error: " + (data.error || "Unknown server error"));
                }
            })
            .catch(err => {
                console.error("[Prediction Error] Inference failed:", err);
            })
            .finally(() => {
                if (btnPredict) {
                    btnPredict.disabled = false;
                    btnPredict.textContent = "Run Machine Learning Risk Assessment";
                }
            });
    }

    function renderPredictionResult(res) {
        const p = res.prediction;

        // Reset badge classes on result box
        if (resultBox) {
            resultBox.className = `result-box ${p.badge_class}`;
        }

        if (resultRiskLabel) resultRiskLabel.textContent = `${p.risk_label} Risk`;
        if (resultConfidenceBadge) resultConfidenceBadge.textContent = `${p.confidence_pct}% Confidence`;

        // Update probability bars
        if (p.probabilities) {
            for (const [tier, pct] of Object.entries(p.probabilities)) {
                if (probBars[tier]) probBars[tier].style.width = `${pct}%`;
                if (probTexts[tier]) probTexts[tier].textContent = `${pct}%`;
            }
        }

        // Update advisories
        if (resultSummary) {
            resultSummary.innerHTML = `<strong>Assessment:</strong> ${p.summary}`;
        }
        if (resultAdvisory) resultAdvisory.textContent = p.advisory;
        if (resultAction) resultAction.textContent = p.action;

        if (resultTimestamp) {
            const now = new Date();
            resultTimestamp.textContent = `Inferred at ${now.toLocaleTimeString()} (DB ID: #${res.db_id || "OK"})`;
        }
    }

    if (predictionForm) {
        predictionForm.addEventListener("submit", function (e) {
            e.preventDefault();
            executePrediction(true);
        });
    }
});
