let qualityChart = null;

// Set current time on header
document.addEventListener("DOMContentLoaded", () => {
    const timeDisplay = document.getElementById("current-time");
    if (timeDisplay) {
        timeDisplay.innerText = "Current Time: " + new Date().toLocaleString();
    }
});

/**
 * SUBMIT DATA TO BACKEND
 */
async function submitData() {
    const data = {
        pH: parseFloat(document.getElementById("pH").value) || 7.0,
        Turbidity: parseFloat(document.getElementById("Turbidity").value) || 0.0,
        DO: parseFloat(document.getElementById("DO").value) || 0.0,
        BOD: parseFloat(document.getElementById("BOD").value) || 0.0,
        Temperature: parseFloat(document.getElementById("Temperature").value) || 20.0
    };

    try {
        const response = await fetch("http://127.0.0.1:8000/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.error) {
            alert("Error: " + result.error);
            return;
        }

        renderDashboard(result);

    } catch (error) {
        console.error("Fetch error:", error);
        alert("Cannot reach the AquaGuard server. Is app.py running?");
    }
}

/**
 * UPDATE UI & CHARTS
 */
function renderDashboard(data) {
    // 1. Update Current Readings Text
    const c = data.current;
    document.getElementById("current-values").innerHTML = `
        <div class="bg-blue-50 p-4 rounded-xl border-l-4 border-water-500 text-gray-700 grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
            <div><b>pH:</b> ${c.pH.toFixed(2)}</div>
            <div><b>Turbidity:</b> ${c.Turbidity.toFixed(2)}</div>
            <div><b>DO:</b> ${c.DO.toFixed(2)} mg/L</div>
            <div><b>BOD:</b> ${c.BOD.toFixed(2)} mg/L</div>
            <div><b>Temp:</b> ${c.Temperature.toFixed(1)}°C</div>
        </div>
    `;

    // 2. Update Alerts
    const alertsBox = document.getElementById("alerts-content");
    alertsBox.innerHTML = "";
    if (data.alerts.length === 0) {
        alertsBox.innerHTML = '<p class="text-green-600 font-medium">✅ Water parameters are within normal ranges.</p>';
    } else {
        data.alerts.forEach(msg => {
            alertsBox.innerHTML += `
                <div class="bg-red-50 border-l-4 border-red-500 p-3 text-red-800 text-sm font-medium">
                    ⚠️ ${msg}
                </div>`;
        });
    }

    // 3. Update Line Chart
    const ctx = document.getElementById("qualityChart").getContext("2d");
    if (qualityChart) qualityChart.destroy();

    qualityChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: data.forecast_times,
            datasets: [{
                label: "Predicted Dissolved Oxygen (mg/L)",
                data: data.forecast.map(row => row.DO),
                borderColor: "#0ea5e9",
                backgroundColor: "rgba(14, 165, 233, 0.1)",
                fill: true,
                tension: 0.4,
                pointRadius: 4,
                pointBackgroundColor: "#0369a1"
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { position: 'top' }
            },
            scales: {
                y: {
                    title: { display: true, text: 'DO Level' },
                    beginAtZero: false
                }
            }
        }
    });
}

/**
 * CHATBOT COMMUNICATION
 */
async function sendMessage() {
    const input = document.getElementById("chat-input");
    const chatBox = document.getElementById("chat-box");
    const message = input.value.trim();

    if (!message) return;

    // Display User Message
    chatBox.innerHTML += `
        <div class="self-end bg-water-600 text-white p-3 rounded-2xl rounded-br-none max-w-[80%] shadow-sm">
            ${message}
        </div>
    `;

    input.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;

    try {
        const response = await fetch("http://127.0.0.1:8000/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: message })
        });

        const data = await response.json();
        const reply = data.reply || "I'm sorry, I couldn't process that.";

        // Display AI Message
        chatBox.innerHTML += `
            <div class="self-start bg-white border border-gray-200 text-gray-800 p-3 rounded-2xl rounded-bl-none max-w-[80%] shadow-sm">
                <b>AquaGuard:</b> ${reply}
            </div>
        `;

    } catch (e) {
        chatBox.innerHTML += `<div class="self-start text-red-500 text-xs">⚠️ AI Connection Error</div>`;
    }

    chatBox.scrollTop = chatBox.scrollHeight;
}