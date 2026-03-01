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
      return React.createElement("div", { className: "muted" }, "Run prediction to see results.");
    }

    const p = props.result.risk_probability;
    const high = p >= 0.5;
    const riskClass = high ? "risk-high" : "risk-ok";

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
          React.createElement("div", { className: "muted" }, "Model Output"),
          React.createElement("div", { className: "risk-value " + riskClass }, p.toFixed(3)),
          React.createElement("div", { className: riskClass }, props.result.risk_label)
        )
      ),
      React.createElement(
        "div",
        { className: "muted", style: { marginTop: "8px" } },
        "Effective FEV1: " + props.result.used_features.fev1.toFixed(3) +
          " L | BMI: " + props.result.used_features.BMI.toFixed(2)
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

  function renderHistory() {
    const history = loadHistory();
    historyTableBody.innerHTML = "";
    if (!history.length) {
      const row = document.createElement("tr");
      row.innerHTML = `<td colspan="5" class="muted">No predictions yet.</td>`;
      historyTableBody.appendChild(row);
      return;
    }
    history.forEach((item) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td>${item.time}</td>
        <td>${item.probability.toFixed(3)}</td>
        <td>${item.label}</td>
        <td>${item.fev1.toFixed(3)}</td>
        <td>${item.bmi.toFixed(2)}</td>
      `;
      historyTableBody.appendChild(tr);
    });
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
