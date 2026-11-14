/**
 * ==============================================================================
 * FLOOD DETECTION SYSTEM - LEAFLET.JS MAP MODULE
 * Renders interactive Punjab OpenStreetMap with dynamic risk markers.
 * ==============================================================================
 */

// Punjab geographic center and default zoom level
const PUNJAB_CENTER = [31.2500, 75.5000];
const DEFAULT_ZOOM = 8;

let floodMap = null;
const stationMarkers = {};

/**
 * Initialize Leaflet OpenStreetMap
 */
function initFloodMap() {
    const mapElement = document.getElementById("punjab-map");
    if (!mapElement) return;

    // Create Map Instance
    floodMap = L.map("punjab-map", {
        center: PUNJAB_CENTER,
        zoom: DEFAULT_ZOOM,
        minZoom: 7,
        maxZoom: 13,
        scrollWheelZoom: true
    });

    // Add OpenStreetMap Tile Layer
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors | Punjab Hydrological Survey',
        maxZoom: 18
    }).addTo(floodMap);

    // Fetch and place all station markers
    loadMapMarkers();

    // Reset map view button listener
    const resetBtn = document.getElementById("btn-reset-map");
    if (resetBtn) {
        resetBtn.addEventListener("click", () => {
            floodMap.flyTo(PUNJAB_CENTER, DEFAULT_ZOOM, { duration: 1.2 });
        });
    }
}

/**
 * Fetch markers from backend API and render on map
 */
function loadMapMarkers() {
    fetch("/api/map-markers")
        .then(res => res.json())
        .then(data => {
            if (data.success && data.markers) {
                data.markers.forEach(station => {
                    createStationMarker(station);
                });
            }
        })
        .catch(err => {
            console.error("[Map Error] Could not load station markers:", err);
        });
}

/**
 * Create a custom HTML divIcon marker with dynamic risk color
 */
function createStationMarker(station) {
    const color = getRiskColor(station.risk_level);

    const customIcon = L.divIcon({
        className: "custom-station-pin",
        html: `<div class="pin-pulse" id="pin-${slugify(station.name)}" style="background-color: ${color};"></div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
        popupAnchor: [0, -12]
    });

    const marker = L.marker([station.latitude, station.longitude], { icon: customIcon }).addTo(floodMap);

    // Create popup HTML
    const popupContent = generatePopupHtml(station);
    marker.bindPopup(popupContent);

    // Save marker reference
    stationMarkers[station.name] = {
        markerInstance: marker,
        data: station
    };

    // When clicked, notify main controller
    marker.on("click", () => {
        if (window.handleMarkerSelected) {
            window.handleMarkerSelected(station.name);
        }
    });
}

/**
 * Generate clean HTML content for Leaflet popup
 */
function generatePopupHtml(station) {
    const badgeClass = getRiskBadgeClass(station.risk_level);

    return `
        <div class="popup-card">
            <div class="popup-header">
                <div class="popup-city">${station.name}</div>
                <div class="popup-coords">${station.latitude}°N, ${station.longitude}°E</div>
            </div>
            <div class="popup-metrics">
                <div class="popup-metric">
                    <span class="p-lbl">24h Rainfall</span>
                    <span class="p-val">${station.rainfall_mm} mm</span>
                </div>
                <div class="popup-metric">
                    <span class="p-lbl">River Level</span>
                    <span class="p-val">${station.river_water_level_m} m</span>
                </div>
                <div class="popup-metric">
                    <span class="p-lbl">Humidity</span>
                    <span class="p-val">${station.humidity_pct}%</span>
                </div>
                <div class="popup-metric">
                    <span class="p-lbl">Risk Tier</span>
                    <span class="badge ${badgeClass}">${station.risk_label}</span>
                </div>
            </div>
            <button class="popup-btn" onclick="selectStationFromPopup('${station.name}')">
                Analyze This Station ➔
            </button>
        </div>
    `;
}

/**
 * Update marker color and popup dynamically when ML prediction changes
 */
function updateMarkerRisk(stationName, riskLevel, riskLabel, confidencePct, inputs) {
    const item = stationMarkers[stationName];
    if (!item) return;

    const color = getRiskColor(riskLevel);
    const pinElem = document.getElementById(`pin-${slugify(stationName)}`);
    if (pinElem) {
        pinElem.style.backgroundColor = color;
    }

    // Update internal data & popup
    item.data.risk_level = riskLevel;
    item.data.risk_label = riskLabel;
    item.data.confidence_pct = confidencePct;
    if (inputs) {
        item.data.rainfall_mm = inputs.rainfall_mm;
        item.data.humidity_pct = inputs.humidity_pct;
        item.data.river_water_level_m = inputs.river_water_level_m;
    }

    item.markerInstance.setPopupContent(generatePopupHtml(item.data));
}

/**
 * Smoothly pan and zoom map to specified location
 */
function centerMapOnStation(stationName) {
    const item = stationMarkers[stationName];
    if (item && floodMap) {
        const coords = [item.data.latitude, item.data.longitude];
        floodMap.flyTo(coords, 10, { duration: 1.0 });
        item.markerInstance.openPopup();
    }
}

/**
 * Global helper callable from popup button
 */
function selectStationFromPopup(stationName) {
    if (window.handleMarkerSelected) {
        window.handleMarkerSelected(stationName);
    }
}

// Color and class mapping utilities
function getRiskColor(level) {
    switch (parseInt(level)) {
        case 0: return "#16a34a"; // Green
        case 1: return "#ca8a04"; // Amber Yellow
        case 2: return "#ea580c"; // Orange
        case 3: return "#dc2626"; // Red
        default: return "#16a34a";
    }
}

function getRiskBadgeClass(level) {
    switch (parseInt(level)) {
        case 0: return "badge-low";
        case 1: return "badge-mod";
        case 2: return "badge-high";
        case 3: return "badge-sev";
        default: return "badge-low";
    }
}

function slugify(text) {
    return text.toString().toLowerCase().replace(/\s+/g, "-").replace(/[^\w\-]+/g, "");
}

// Initialize on DOM load
document.addEventListener("DOMContentLoaded", initFloodMap);
