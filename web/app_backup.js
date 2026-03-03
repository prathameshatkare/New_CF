(function () {
  const healthEl = document.getElementById("health");
  const fev1ModeEl = document.getElementById("fev1-mode");
  const fev1FieldsEl = document.getElementById("fev1-fields");
  const autoBmiEl = document.getElementById("auto-bmi");
  const bmiManualWrap = document.getElementById("bmi-manual-wrap");
  const formEl = document.getElementById("predict-form");
  const historyTableBody = document.querySelector("#history-table tbody");
  const tabButtons = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const themeToggle = document.getElementById("theme-toggle");
  
  // Theme Toggle Functionality
  const THEME_KEY = "cf_theme_preference";
  
  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem(THEME_KEY, theme);
    themeToggle.textContent = theme === "dark" ? "☀️" : "🌙";
  }
  
  function initTheme() {
    const savedTheme = localStorage.getItem(THEME_KEY) || "light";
    setTheme(savedTheme);
  }
  
  themeToggle.addEventListener("click", () => {
    const currentTheme = document.documentElement.getAttribute("data-theme");
    const newTheme = currentTheme === "dark" ? "light" : "dark";
    setTheme(newTheme);
  });
  
  initTheme();
  
  // Real-time Form Validation
  function validateField(inputEl, validator) {
    const label = inputEl.closest("label");
    const errorMsg = label.querySelector(".validation-message");
    const isValid = validator(inputEl.value);
    
    if (isValid) {
      inputEl.classList.remove("invalid");
      inputEl.classList.add("valid");
      if (errorMsg) errorMsg.style.display = "none";
    } else {
      inputEl.classList.remove("valid");
      inputEl.classList.add("invalid");
      if (errorMsg) errorMsg.style.display = "block";
    }
    
    return isValid;
  }
  
  // Age validation
  const ageInput = document.getElementById("age");
  if (ageInput) {
    ageInput.addEventListener("blur", () => {
      validateField(ageInput, (val) => {
        const num = Number(val);
        return num >= 0 && num <= 120;
      });
    });
  }
  
  // Height validation
  const heightInput = document.getElementById("height");
  if (heightInput) {
    heightInput.addEventListener("blur", () => {
      validateField(heightInput, (val) => {
        const num = Number(val);
        return num >= 40 && num <= 250;
      });
    });
  }
  
  // Weight validation
  const weightInput = document.getElementById("weight");
  if (weightInput) {
    weightInput.addEventListener("blur", () => {
      validateField(weightInput, (val) => {
        const num = Number(val);
        return num >= 5 && num <= 300;
      });
    });
  }

  let latestResult = null;
  const HISTORY_KEY = "cf_prediction_history_v2";
  const reactRoot = ReactDOM.createRoot(document.getElementById("react-root"));

  function switchTab(tabName) {
    tabButtons.forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === tabName));
    tabContents.forEach((panel) => panel.classList.toggle("active", panel.id === "tab-" + tabName));
  }

  function renderFev1Fields() {
    const mode = fev1ModeEl.value;
    if (mode === "liters") {
      fev1FieldsEl.innerHTML = `
        <label>FEV1 (L)
          <input type="number" id="fev1-l" min="0" max="12" step="0.01" value="2.5" required />
        </label>
      `;
      return;
    }
    if (mode === "ml") {
      fev1FieldsEl.innerHTML = `
        <label>FEV1 (mL)
          <input type="number" id="fev1-ml" min="0" max="12000" step="1" value="2500" required />
        </label>
      `;
      return;
    }
    fev1FieldsEl.innerHTML = `
      <label>FEV1 % predicted
        <input type="number" id="fev1-pct" min="0" max="300" step="0.1" value="80" required />
      </label>
      <label>Predicted FEV1 from report (L)
        <input type="number" id="fev1-pred-l" min="0" max="12" step="0.01" value="3.0" required />
      </label>
    `;
  }

  function getFev1Liters() {
    const mode = fev1ModeEl.value;
    if (mode === "liters") return Number(document.getElementById("fev1-l").value);
    if (mode === "ml") return Number(document.getElementById("fev1-ml").value) / 1000.0;
    const pct = Number(document.getElementById("fev1-pct").value);
    const predL = Number(document.getElementById("fev1-pred-l").value);
    return predL * (pct / 100.0);
  }

  function RiskResultCard(props) {
    if (!props.result) {
      return React.createElement("div", { className: "muted", style: { padding: "20px", textAlign: "center" } }, 
        "⬅️ Enter patient data and click predict to see risk analysis");
    }

    const p = props.result.risk_probability;
    const high = p >= 0.5;
    const riskClass = high ? "risk-high" : "risk-ok";
    const riskIcon = high ? "⚠️" : "✅";
    
    // Determine risk level text
    let riskLevel = "";
    let riskColor = "";
    if (p < 0.3) {
      riskLevel = "Low Risk";
      riskColor = "var(--ok)";
    } else if (p < 0.5) {
      riskLevel = "Moderate Risk";
      riskColor = "var(--warning)";
    } else if (p < 0.7) {
      riskLevel = "High Risk";
      riskColor = "var(--danger)";
    } else {
      riskLevel = "Very High Risk";
      riskColor = "var(--danger-dark)";
    }

    return React.createElement(
      "div",
      { className: "risk-card" },
      React.createElement(
        "div",
        { className: "gauge-wrap" },
        React.createElement(
          "div",
          { className: "gauge", style: { "--value": (p * 100).toFixed(1) } },
          React.createElement("span", null, (p * 100).toFixed(1) + "%")
        ),
        React.createElement(
          "div",
          null,
          React.createElement(
            "div", 
            { className: "muted", style: { fontSize: "0.85rem", fontWeight: "600", textTransform: "uppercase", letterSpacing: "0.05em" } }, 
            "AI Model Prediction"
          ),
          React.createElement("div", { className: "risk-value " + riskClass, style: { color: riskColor } }, p.toFixed(3)),
          React.createElement(
            "div", 
            { className: riskClass, style: { 
              fontSize: "1.1rem", 
              fontWeight: "700", 
              marginTop: "8px",
              display: "flex",
              alignItems: "center",
              gap: "8px"
            }}, 
            riskIcon + " " + props.result.risk_label
          ),
          React.createElement(
            "div",
            { style: { 
              marginTop: "12px", 
              padding: "12px", 
              background: "rgba(59, 130, 246, 0.08)", 
              borderRadius: "12px",
              borderLeft: "3px solid " + riskColor
            }},
            React.createElement("div", { style: { fontWeight: "600", marginBottom: "6px" } }, "📊 Risk Assessment"),
            React.createElement("div", { className: "muted" }, 
              React.createElement("strong", null, riskLevel), 
              ": Based on the federated learning model trained across multiple institutions with privacy-preserving techniques."
            )
          )
        )
      ),
      React.createElement(
        "div",
        { style: { 
          marginTop: "16px", 
          padding: "14px", 
          background: "linear-gradient(135deg, rgba(6, 182, 212, 0.08), rgba(59, 130, 246, 0.08))",
          borderRadius: "12px"
        }},
        React.createElement(
          "div",
          { style: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" } },
          React.createElement(
            "div",
            null,
            React.createElement("div", { className: "muted", style: { fontSize: "0.75rem", fontWeight: "600", textTransform: "uppercase" } }, "FEV1 (Lung Function)"),
            React.createElement("div", { style: { fontSize: "1.2rem", fontWeight: "700", color: "var(--text)" } }, 
              props.result.used_features.fev1.toFixed(3) + " L"
            )
          ),
          React.createElement(
            "div",
            null,
            React.createElement("div", { className: "muted", style: { fontSize: "0.75rem", fontWeight: "600", textTransform: "uppercase" } }, "BMI (Nutritional Status)"),
            React.createElement("div", { style: { fontSize: "1.2rem", fontWeight: "700", color: "var(--text)" } }, 
              props.result.used_features.BMI.toFixed(2)
            )
          )
        )
      )
    );
  }

  function renderResult() {
    reactRoot.render(React.createElement(RiskResultCard, { result: latestResult }));
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  function saveHistory(items) {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(items.slice(0, 10)));
  }

  let historyChart = null;
  
  function initHistoryChart(history) {
    const ctx = document.getElementById("history-chart");
    if (!ctx) return;
    
    // Reverse to show chronological order (oldest first)
    const reversedHistory = [...history].reverse();
    
    const data = {
      labels: reversedHistory.map(item => item.time),
      datasets: [{
        label: "CF Risk Probability",
        data: reversedHistory.map(item => item.probability),
        borderColor: "rgb(59, 130, 246)",
        backgroundColor: "rgba(59, 130, 246, 0.1)",
        borderWidth: 3,
        fill: true,
        tension: 0.4,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBackgroundColor: reversedHistory.map(item => 
          item.probability >= 0.5 ? "rgb(239, 68, 68)" : "rgb(16, 185, 129)"
        ),
        pointBorderColor: "#fff",
        pointBorderWidth: 2,
      }]
    };
    
    const config = {
      type: "line",
      data: data,
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          intersect: false,
          mode: "index",
        },
        plugins: {
          legend: {
            display: true,
            position: "top",
            labels: {
              usePointStyle: true,
              padding: 15,
              font: {
                size: 12,
                weight: "600",
              },
            },
          },
          tooltip: {
            backgroundColor: "rgba(15, 23, 42, 0.95)",
            titleColor: "#fff",
            bodyColor: "#fff",
            borderColor: "rgba(59, 130, 246, 0.5)",
            borderWidth: 1,
            padding: 12,
            displayColors: false,
            callbacks: {
              label: function(context) {
                const value = (context.parsed.y * 100).toFixed(1) + "%";
                const item = reversedHistory[context.dataIndex];
                return [
                  `Risk: ${value}`,
                  `FEV1: ${item.fev1.toFixed(3)} L`,
                  `BMI: ${item.bmi.toFixed(2)}`
                ];
              },
              title: function(items) {
                return "Time: " + items[0].label;
              },
            },
          },
        },
        scales: {
          y: {
            min: 0,
            max: 1,
            ticks: {
              format: { style: "percent" },
              color: getComputedStyle(document.documentElement).getPropertyValue("--text-light"),
            },
            grid: {
              color: "rgba(59, 130, 246, 0.1)",
            },
          },
          x: {
            display: false,
          },
        },
      },
    };
    
    if (historyChart) {
      historyChart.destroy();
    }
    
    historyChart = new Chart(ctx, config);
  }
  
  function renderHistory() {
    const history = loadHistory();
    historyTableBody.innerHTML = "";
    if (!history.length) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="5" class="muted">No predictions yet. Start by entering patient data!</td>`;
      historyTableBody.appendChild(row);
      
      if (historyChart) {
        historyChart.destroy();
        historyChart = null;
      }
      return;
    }
    history.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.time}</td>
        <td>${item.probability.toFixed(3)}</td>
        <td><span class="badge ${item.probability >= 0.5 ? 'badge-danger' : 'badge-success'}">${item.label}</span></td>
        <td>${item.fev1.toFixed(3)}</td>
        <td>${item.bmi.toFixed(2)}</td>
      `;
      historyTableBody.appendChild(tr);
    });
    
    // Initialize or update chart
    initHistoryChart(history);
  }

  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      healthEl.textContent = "Model ready | Features: " + data.features.join(", ");
      healthEl.style.background = "rgba(255,255,255,0.24)";
    } catch (err) {
      healthEl.textContent = "API unavailable";
    }
  }

  autoBmiEl.addEventListener("change", function () {
    bmiManualWrap.classList.toggle("hidden", autoBmiEl.checked);
  });

  fev1ModeEl.addEventListener("change", renderFev1Fields);
  tabButtons.forEach((btn) => btn.addEventListener("click", () => switchTab(btn.dataset.tab)));

  renderFev1Fields();
  renderResult();
  renderHistory();
  checkHealth();

  // What-If Analysis Functionality
  const whatIfSection = document.getElementById("what-if-section");
  const toggleWhatIfBtn = document.getElementById("toggle-what-if");
  const fev1Slider = document.getElementById("fev1-slider");
  const bmiSlider = document.getElementById("bmi-slider");
  const fev1SliderValue = document.getElementById("fev1-slider-value");
  const bmiSliderValue = document.getElementById("bmi-slider-value");
  const adjustedRiskDisplay = document.getElementById("adjusted-risk");
  
  let currentPrediction = null;
  
  if (toggleWhatIfBtn) {
    toggleWhatIfBtn.addEventListener("click", () => {
      if (whatIfSection.style.display === "none") {
        whatIfSection.style.display = "block";
        toggleWhatIfBtn.textContent = "✖️ Close What-If";
        if (latestResult && fev1Slider && bmiSlider) {
          fev1Slider.value = latestResult.used_features.fev1.toFixed(1);
          bmiSlider.value = latestResult.used_features.BMI.toFixed(1);
          fev1SliderValue.textContent = fev1Slider.value + " L";
          bmiSliderValue.textContent = bmiSlider.value;
        }
      } else {
        whatIfSection.style.display = "none";
        toggleWhatIfBtn.textContent = "✨ Try What-If Analysis";
      }
    });
  }
  
  if (fev1Slider) {
    fev1Slider.addEventListener("input", (e) => {
      fev1SliderValue.textContent = e.target.value + " L";
      updateWhatIfPrediction();
    });
  }
  
  if (bmiSlider) {
    bmiSlider.addEventListener("input", (e) => {
      bmiSliderValue.textContent = e.target.value;
      updateWhatIfPrediction();
    });
  }
  
  function updateWhatIfPrediction() {
    if (!currentPrediction || !fev1Slider || !bmiSlider) return;
    
    const newFev1 = parseFloat(fev1Slider.value);
    const newBmi = parseFloat(bmiSlider.value);
    
    // Simple linear approximation based on feature importance
    const baseRisk = currentPrediction.risk_probability;
    const originalFev1 = currentPrediction.used_features.fev1;
    const originalBmi = currentPrediction.used_features.BMI;
    
    // FEV1 has inverse relationship with risk (lower FEV1 = higher risk)
    const fev1Change = (originalFev1 - newFev1) / originalFev1;
    // BMI has moderate relationship
    const bmiChange = (newBmi - originalBmi) / originalBmi;
    
    // Adjusted risk calculation (simplified model)
    const adjustedRisk = Math.max(0, Math.min(1, baseRisk + (fev1Change * 0.4) + (bmiChange * 0.15)));
    
    adjustedRiskDisplay.textContent = (adjustedRisk * 100).toFixed(1) + "%";
    
    // Color coding
    if (adjustedRisk < 0.3) {
      adjustedRiskDisplay.style.color = "var(--ok)";
    } else if (adjustedRisk < 0.5) {
      adjustedRiskDisplay.style.color = "var(--warning)";
    } else {
      adjustedRiskDisplay.style.color = "var(--danger)";
    }
  }

  formEl.addEventListener("submit", async function (e) {
    e.preventDefault();
    const age = Number(document.getElementById("age").value);
    const sex = document.getElementById("sex").value;
    const height = Number(document.getElementById("height").value);
    const weight = Number(document.getElementById("weight").value);
    const fev1_l = getFev1Liters();

    const payload = {
      age,
      sex,
      height,
      weight,
      fev1_l,
      bmi: autoBmiEl.checked ? null : Number(document.getElementById("bmi-manual").value),
    };

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || "Prediction failed");
      }

      latestResult = data;
      currentPrediction = data; // Store for what-if analysis
      
      // Update what-if sliders with current values
      if (fev1Slider && bmiSlider) {
        fev1Slider.value = data.used_features.fev1.toFixed(1);
        bmiSlider.value = data.used_features.BMI.toFixed(1);
        fev1SliderValue.textContent = data.used_features.fev1.toFixed(2) + " L";
        bmiSliderValue.textContent = data.used_features.BMI.toFixed(2);
      }
      
      const history = loadHistory();
      history.unshift({
        time: new Date().toLocaleTimeString(),
        probability: data.risk_probability,
        label: data.risk_label,
        fev1: data.used_features.fev1,
        bmi: data.used_features.BMI,
      });
      saveHistory(history);
      renderHistory();
      renderResult();
      switchTab("prediction");
    } catch (err) {
      latestResult = {
        risk_probability: 0,
        risk_label: String(err.message || err),
        used_features: {
          fev1: fev1_l,
          BMI: payload.bmi || payload.weight / ((payload.height / 100) ** 2),
        },
      };
      renderResult();
      switchTab("prediction");
    }
  });
})();
