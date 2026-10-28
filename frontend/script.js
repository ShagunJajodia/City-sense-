const API_URL = (window.location.origin && window.location.origin.startsWith("http")) ? window.location.origin : "http://127.0.0.1:8000";
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss://" : "ws://";
const WS_HOST = window.location.host || "127.0.0.1:8000";
const WS_URL = `${WS_PROTOCOL}${WS_HOST}/ws/live_simulate`;
let map;
let markers = {};
let allZonesData = [];
let liveWs = null;
let routingControl = null;

let routeStart = null;
let routeEnd = null;
let currentGlobalHour = 12; // 0 to 24

function switchTab(tabId, ev) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    
    const targetTab = document.getElementById(tabId);
    if (targetTab) targetTab.classList.add('active');

    const clickedBtn = (ev && ev.currentTarget) || (window.event && window.event.currentTarget) || document.querySelector(`[onclick*="${tabId}"]`);
    if (clickedBtn) clickedBtn.classList.add('active');
    
    if (tabId === 'tab-main' && map) setTimeout(() => map.invalidateSize(), 100);
}

document.addEventListener("DOMContentLoaded", async () => {
    initMap();
    await fetchMetrics();
    await fetchZonesAndInit();
    
    document.getElementById("live-tick-btn").addEventListener("click", toggleLiveSimulation);
    document.getElementById("time-slider").addEventListener("input", handleTimeSlider);
    document.getElementById("calc-route-btn").addEventListener("click", calculateSafeRoute);
});

function initMap() {
    map = L.map('map-container').setView([12.9716, 77.5946], 11);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ',
        maxZoom: 16
    }).addTo(map);
    L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}', {
        attribution: '&copy; Esri',
        maxZoom: 16
    }).addTo(map);
}

// Map Time Slider (0 to 24) to string representation and simulate Traffic impact
function formatHour(val) {
    val = parseInt(val);
    if(val === 0 || val === 24) return "12 AM (Midnight)";
    if(val === 12) return "12 PM (Noon)";
    if(val < 12) return `${val} AM`;
    return `${val - 12} PM`;
}

function getTrafficForTime(baseTraffic, hour) {
    hour = parseInt(hour);
    // Rush hours: 8-10 AM, 5-8 PM (17-20)
    let isRush = (hour >= 8 && hour <= 10) || (hour >= 17 && hour <= 20);
    let isNight = (hour <= 5 || hour >= 23);
    
    if (isRush) {
        if (baseTraffic === "Low") return "Normal";
        if (baseTraffic === "Normal") return "High";
        return "Severe";
    }
    if (isNight) {
        if (baseTraffic === "Severe" || baseTraffic === "High") return "Normal";
        return "Low";
    }
    return baseTraffic; // Midday
}

async function handleTimeSlider(e) {
    currentGlobalHour = e.target.value;
    document.getElementById("current-slider-time").innerText = formatHour(currentGlobalHour);
    
    // Recalculate predictions for all zones based on new time-impacted traffic!
    for(const z of allZonesData) {
        let simulatedTraffic = getTrafficForTime(z.base_traffic || "Normal", currentGlobalHour);
        
        const payload = {
            Zone: z.name,
            Traffic: simulatedTraffic,
            Weather: "Clear", // could vary this by time too
            Road_Condition: "Good",
            Pollution_AQI: z.base_aqi || 80,
            Water_Quality: z.base_water || "Good",
            Transport_Connection: z.transport || 8
        };
        
        try {
            const pRes = await fetch(`${API_URL}/predict`, {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload)
            });
            const pData = await pRes.json();
            
            let color = '#10b981'; // Low
            if (pData.accident_level === 'Normal') color = '#f59e0b';
            if (pData.accident_level === 'High') color = '#ef4444';
            
            if (markers[z.name]) {
                markers[z.name].setStyle({ fillColor: color });
            }
        } catch(err) {}
    }
}

async function fetchZonesAndInit() {
    try {
        const res = await fetch(`${API_URL}/zones`);
        const data = await res.json();
        allZonesData = data.zones;
        
        for(const z of allZonesData) {
            const payload = {
                Zone: z.name,
                Traffic: z.base_traffic || "Normal",
                Weather: "Clear",
                Road_Condition: "Good",
                Pollution_AQI: z.base_aqi || 80,
                Water_Quality: z.base_water || "Good",
                Transport_Connection: z.transport || 8
            };
            
            let color = '#9ca3af';
            try {
                const pRes = await fetch(`${API_URL}/predict`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                const pData = await pRes.json();
                if (pData.accident_level === 'Low') color = '#10b981';
                else if (pData.accident_level === 'Normal') color = '#f59e0b';
                else if (pData.accident_level === 'High') color = '#ef4444';
            } catch(e) {}

            const marker = L.circleMarker([z.lat, z.lon], {
                color: 'transparent',
                fillColor: color,
                fillOpacity: 0.3,
                radius: 40,
                weight: 0
            }).addTo(map);
            
            // Standard Click logic vs Routing Click logic
            marker.on('click', () => handleZoneClick(z));
            
            // Context menu logic for Routing (Right Click / Double click logic fallback)
            marker.on('contextmenu', (e) => handleRoutingSelect(z));

            markers[z.name] = marker;
        }
    } catch (e) {
        console.error("Map Zones Error", e);
    }
}

function handleRoutingSelect(zone) {
    if(!routeStart) {
        routeStart = zone;
        document.getElementById("route-start").innerText = zone.name;
        document.getElementById("route-start").classList.remove("empty");
    } else if (!routeEnd) {
        routeEnd = zone;
        document.getElementById("route-end").innerText = zone.name;
        document.getElementById("route-end").classList.remove("empty");
    } else {
        // Reset
        routeStart = zone;
        routeEnd = null;
        document.getElementById("route-start").innerText = zone.name;
        document.getElementById("route-end").innerText = "End Zone";
        document.getElementById("route-end").classList.add("empty");
        if(routingControl) map.removeControl(routingControl);
    }
}

function calculateSafeRoute() {
    if(!routeStart || !routeEnd) {
        alert("Please right-click on two different map markers to set Start and End origins.");
        return;
    }
    
    if(routingControl) {
        map.removeControl(routingControl);
    }
    
    // We create an A-to-B route. To theoretically "dodge" red zones, 
    // an advanced algorithm would plot waypoints. For this UI mockup MVP, 
    // we use Leaflet Routing standard between the two specific points.
    routingControl = L.Routing.control({
        waypoints: [
            L.latLng(routeStart.lat, routeStart.lon),
            L.latLng(routeEnd.lat, routeEnd.lon)
        ],
        show: true,
        routeWhileDragging: false,
        lineOptions: {
            styles: [{color: '#10b981', weight: 4}] // Draw safe green route line
        },
        createMarker: function() { return null; } // Hide extra markers
    }).addTo(map);
}

async function handleZoneClick(zone) {
    let simulatedTraffic = getTrafficForTime(zone.base_traffic || "Normal", currentGlobalHour);
    
    const payload = {
        Zone: zone.name,
        Traffic: simulatedTraffic,
        Weather: "Clear",
        Road_Condition: "Good",
        Pollution_AQI: zone.base_aqi || 80,
        Water_Quality: zone.base_water || "Good",
        Transport_Connection: zone.transport || 8
    };

    try {
        const res = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        // Show Core Panels
        document.getElementById("livability-panel").style.display = 'block';
        document.getElementById("advisory-panel").style.display = 'block';
        
        // Set metrics
        document.getElementById("liv-score").innerText = data.livability_index;
        const badge = document.getElementById("acc-risk-badge");
        badge.innerText = data.accident_level.toUpperCase();
        badge.className = `badge ${data.accident_level.toLowerCase()}`;
        
        // Set XAI Text
        document.getElementById("xai-text").innerText = data.xai_reason;
        
        // Set Advisory LLM Text
        document.getElementById("adv-text").innerText = data.ai_advisory;
        
        // Details
        document.getElementById("liv-details").innerHTML = `
            <div><span>Time/Traffic</span><span>${simulatedTraffic} at ${formatHour(currentGlobalHour)}</span></div>
            <div><span>Pollution AQI</span><span>${payload.Pollution_AQI}</span></div>
            <div><span>Water Quality</span><span>${payload.Water_Quality}</span></div>
            <div><span>Transport Rating</span><span>${payload.Transport_Connection}/10</span></div>
        `;
    } catch (e) {
        console.error(e);
    }
}

// Websocket Logic Toggle
function toggleLiveSimulation() {
    const btn = document.getElementById("live-tick-btn");
    const status = document.getElementById("sim-status");
    
    if (liveWs) {
        // Disconnect
        liveWs.close();
        liveWs = null;
        btn.innerText = "🚀 Connect Stream (WebSocket)";
        status.innerText = "● Standby";
        status.className = "status-indicator";
    } else {
        // Connect WebSocket
        btn.innerText = "🛑 Disconnect Stream";
        status.innerText = "● WebSocket Streaming Data...";
        status.className = "status-indicator running";
        
        liveWs = new WebSocket(WS_URL);
        
        liveWs.onmessage = function(event) {
            const data = JSON.parse(event.data);
            data.simulations.forEach(sim => {
                const marker = markers[sim.zone_info.name];
                if (marker) {
                    let color = '#10b981'; // green = Low
                    if (sim.prediction.accident_level === 'Normal') color = '#f59e0b'; // yellow
                    if (sim.prediction.accident_level === 'High') color = '#ef4444'; // red
                    
                    // Flash marker to simulate active hit
                    marker.setStyle({ color: '#fff', fillColor: color, fillOpacity: 0.8, weight: 2 });
                    setTimeout(() => {
                        marker.setStyle({ color: 'transparent', fillOpacity: 0.3, weight: 0 });
                    }, 500);
                }
            });
        };
        
        liveWs.onerror = function(error) {
            console.error("WS Error", error);
            toggleLiveSimulation(); // rollback UI
        };
    }
}

async function fetchMetrics() {
    try {
        const res = await fetch(`${API_URL}/metrics`);
        const data = await res.json();
        renderCharts(data);
    } catch(e) {
        console.error("Metrics render error", e);
    }
}

function renderCharts(data) {
    const models = data.models;
    const labels = models.map(m => m.name);
    
    Chart.defaults.color = "#a1a1aa";
    Chart.defaults.font.family = "Inter";

    // 1. Accuracy
    new Chart(document.getElementById('accChart'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Accuracy',
                data: models.map(m => m.accuracy),
                backgroundColor: ['rgba(255, 255, 255, 0.8)', 'rgba(16, 185, 129, 0.6)', 'rgba(245, 158, 11, 0.6)'],
                borderColor: ['#fff', '#10b981', '#f59e0b'],
                borderWidth: 1
            }]
        },
        options: { responsive: true, maintainAspectRatio: false, scales: { y: { min: 0.5, max: 1.0 } } }
    });

    // 2. Precision/Recall/F1
    new Chart(document.getElementById('prfChart'), {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                { label: 'Precision', data: models.map(m => m.precision), backgroundColor: 'rgba(255,255,255,0.8)' },
                { label: 'Recall', data: models.map(m => m.recall), backgroundColor: '#10b981' },
                { label: 'F1 Score', data: models.map(m => m.f1), backgroundColor: '#f59e0b' }
            ]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // 3. Time
    new Chart(document.getElementById('timeChart'), {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Time (s)',
                data: models.map(m => m.training_time),
                borderColor: '#ffffff',
                backgroundColor: 'rgba(255, 255, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: { responsive: true, maintainAspectRatio: false }
    });

    // 4. CM
    const bestModel = models.reduce((prev, current) => (prev.accuracy > current.accuracy) ? prev : current);
    const cm = bestModel.confusion_matrix;
    
    const grid = document.getElementById("cm-grid");
    document.getElementById("cm-card").querySelector("h4").innerText = `Confusion Matrix (${bestModel.name})`;
    grid.innerHTML = '';
    
    const maxVal = Math.max(...cm.flat());
    for(let r=0; r<cm.length; r++) {
        for(let c=0; c<cm[r].length; c++) {
            const val = cm[r][c];
            const intensity = Math.max(0.1, val / maxVal);
            const cell = document.createElement("div");
            cell.className = "cm-cell";
            cell.style.backgroundColor = `rgba(255, 255, 255, ${intensity})`;
            cell.style.color = intensity > 0.5 ? '#000' : '#fff';
            cell.innerText = val;
            grid.appendChild(cell);
        }
    }
}
