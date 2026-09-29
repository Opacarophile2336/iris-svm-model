import { useState } from 'react';
import { getInsights, type PredictRequest } from '../../api/prediction';

const SPECIES_COLORS: Record<string, string> = {
  'Iris setosa': '#4E79A7',
  'Iris versicolor': '#F28E2B',
  'Iris virginica': '#59A14F',
};

const FEATURE_NAMES: Record<string, string> = {
  sepal_length: 'Sepal Length',
  sepal_width: 'Sepal Width',
  petal_length: 'Petal Length',
  petal_width: 'Petal Width',
};

const KERNEL_COLORS: Record<string, string> = {
  rbf: '#4E79A7',
  linear: '#F28E2B',
  poly: '#59A14F',
  sigmoid: '#E15759',
  mlp: '#76B7B2',
};

const MODEL_DESCRIPTIONS: Record<string, string> = {
  rbf: 'Gaussian RBF kernel projects features into infinite dimensions for localized boundary clusters.',
  linear: 'Linear hyperplane separation identifies global maximum-margin boundary.',
  poly: '3rd-order polynomial boundary captures curved parabolic intersections.',
  sigmoid: 'Hyperbolic tangent mapping models S-curve boundary transitions.',
  mlp: 'Multi-layer perceptron neural network (64, 32) with ReLU and Softmax distribution.',
};

const SAMPLES = [
  { label: 'Canonical Setosa', sepal_length: 5.0, sepal_width: 3.4, petal_length: 1.5, petal_width: 0.2 },
  { label: 'Typical Versicolor', sepal_length: 5.9, sepal_width: 2.8, petal_length: 4.2, petal_width: 1.3 },
  { label: 'Robust Virginica', sepal_length: 6.5, sepal_width: 3.0, petal_length: 5.5, petal_width: 2.0 },
  { label: 'Boundary Overlap Sample', sepal_length: 6.0, sepal_width: 2.7, petal_length: 4.9, petal_width: 1.8 },
];

export default function ModelInsights() {
  const [inputs, setInputs] = useState<PredictRequest>({
    sepal_length: 5.8,
    sepal_width: 2.7,
    petal_length: 4.1,
    petal_width: 1.2,
  });

  const [insightData, setInsightData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleAnalyze = async () => {
    setError('');
    setLoading(true);
    try {
      const res = await getInsights(inputs);
      setInsightData(res.data);
    } catch (err: any) {
      setError(err.response?.data?.detail || 'Failed to generate model insights.');
    } finally {
      setLoading(false);
    }
  };

  const loadSample = (sample: typeof SAMPLES[0]) => {
    setInputs({
      sepal_length: sample.sepal_length,
      sepal_width: sample.sepal_width,
      petal_length: sample.petal_length,
      petal_width: sample.petal_width,
    });
    setInsightData(null);
  };

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">💡 Model Insights & Botanical Diagnostics</h1>
        <p className="page-subtitle">
          Explainable ML diagnostics connecting SVM decision boundaries and Deep Learning neural representations with empirical botanical distributions
        </p>
      </div>

      {/* Preset bar */}
      <div className="card" style={{ marginBottom: '1.2rem' }}>
        <div className="card-header-row">
          <h2 className="card-title">Morphological Input Parameters</h2>
          <div className="preset-buttons">
            <span className="text-muted" style={{ fontSize: '0.85rem' }}>Presets: </span>
            {SAMPLES.map((s, idx) => (
              <button key={idx} className="btn-example" onClick={() => loadSample(s)}>
                {s.label}
              </button>
            ))}
          </div>
        </div>

        <div className="insight-input-grid">
          <div className="form-group">
            <label className="form-label">Sepal Length (cm)</label>
            <input
              type="number"
              step="0.1"
              className="form-input"
              value={inputs.sepal_length}
              onChange={e => setInputs(prev => ({ ...prev, sepal_length: parseFloat(e.target.value) || 0 }))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Sepal Width (cm)</label>
            <input
              type="number"
              step="0.1"
              className="form-input"
              value={inputs.sepal_width}
              onChange={e => setInputs(prev => ({ ...prev, sepal_width: parseFloat(e.target.value) || 0 }))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Petal Length (cm)</label>
            <input
              type="number"
              step="0.1"
              className="form-input"
              value={inputs.petal_length}
              onChange={e => setInputs(prev => ({ ...prev, petal_length: parseFloat(e.target.value) || 0 }))}
            />
          </div>
          <div className="form-group">
            <label className="form-label">Petal Width (cm)</label>
            <input
              type="number"
              step="0.1"
              className="form-input"
              value={inputs.petal_width}
              onChange={e => setInputs(prev => ({ ...prev, petal_width: parseFloat(e.target.value) || 0 }))}
            />
          </div>
        </div>

        {error && <div className="alert-error" style={{ marginTop: '0.8rem' }}>{error}</div>}

        <div style={{ marginTop: '0.8rem' }}>
          <button className="btn-primary" onClick={handleAnalyze} disabled={loading}>
            {loading ? '🔬 Computing Diagnostics...' : '💡 Generate Model Insights'}
          </button>
        </div>
      </div>

      {/* Insights Results */}
      {insightData && (
        <div className="insights-results-container">
          {/* Top Summary Card */}
          <div className="card-grid">
            <div className="stat-card blue">
              <div className="stat-value">{insightData.highest_confidence?.confidence}%</div>
              <div className="stat-label">Peak Confidence ({insightData.highest_confidence?.kernel_display})</div>
            </div>
            <div className="stat-card green">
              <div className="stat-value">{insightData.highest_confidence?.predicted_species?.replace('Iris ', '')}</div>
              <div className="stat-label">Primary Classification</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{insightData.insights?.consensus?.toUpperCase()}</div>
              <div className="stat-label">Classifier Consensus Index</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{insightData.insights?.closest_botanical_match?.replace('Iris ', '')}</div>
              <div className="stat-label">Morphological Match</div>
            </div>
          </div>

          {/* 5-Classifier Multi-Model Insights Matrix */}
          <div className="card" style={{ marginTop: '1.2rem' }}>
            <div className="card-header-row">
              <h2 className="card-title">5-Classifier Multi-Model Insights Matrix</h2>
              <span className="consensus-tag" style={{ background: '#EBF0F8', color: '#1F3864', fontWeight: 600 }}>
                {insightData.results ? Object.keys(insightData.results).length : 5} Models Evaluated
              </span>
            </div>
            <p className="text-muted" style={{ marginBottom: '1rem' }}>
              Comparative breakdown showing how each classifier family (Linear Hyperplane, Non-Linear SVM Kernels, and Deep Learning Neural Network) evaluates this specimen.
            </p>

            <div
              className="kernel-comparison-grid"
              style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))' }}
            >
              {(['rbf', 'linear', 'poly', 'sigmoid', 'mlp'] as const).map((k) => {
                const kRes = insightData.results?.[k];
                if (!kRes) return null;
                const isTop = k === insightData.highest_confidence?.kernel;
                const color = KERNEL_COLORS[k] || '#4E79A7';
                const explanation = insightData.insights?.model_explanations?.[k] || MODEL_DESCRIPTIONS[k];

                return (
                  <div
                    key={k}
                    className={`kernel-result-box ${isTop ? 'highlighted-kernel' : ''}`}
                    style={{ borderTop: `3px solid ${color}`, display: 'flex', flexDirection: 'column' }}
                  >
                    <div className="kernel-box-header">
                      <span className="kernel-box-title" style={{ color, fontWeight: 700 }}>
                        {kRes.kernel_display}
                      </span>
                      {isTop && <span className="best-badge">Highest Conf.</span>}
                    </div>

                    <div
                      className="kernel-box-prediction"
                      style={{ color: SPECIES_COLORS[kRes.predicted_species] || '#212529', margin: '6px 0', fontSize: '1.05rem' }}
                    >
                      {kRes.predicted_species}
                    </div>

                    <div className="kernel-metric-row">
                      <span className="metric-name">Confidence:</span>
                      <span className="metric-val"><strong>{kRes.confidence}%</strong></span>
                    </div>
                    <div className="kernel-metric-row">
                      <span className="metric-name">Validated Acc:</span>
                      <span className="metric-val">{kRes.accuracy}%</span>
                    </div>

                    {/* Miniature probability bars */}
                    <div className="mini-prob-bars" style={{ marginTop: '8px' }}>
                      {Object.entries(kRes.probabilities || {}).map(([sp, prob]: [string, any]) => (
                        <div key={sp} className="mini-prob-row">
                          <span className="mini-prob-label">{sp}</span>
                          <div className="mini-bar-track">
                            <div
                              className="mini-bar-fill"
                              style={{
                                width: `${prob}%`,
                                backgroundColor: SPECIES_COLORS[`Iris ${sp}`] || color,
                              }}
                            />
                          </div>
                          <span className="mini-prob-num">{prob}%</span>
                        </div>
                      ))}
                    </div>

                    {/* Model-specific architectural explanation */}
                    <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px solid var(--border)', fontSize: '0.74rem', color: 'var(--text-muted)', lineHeight: '1.35' }}>
                      {explanation}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Academic Diagnostics Breakdown */}
          <div className="card" style={{ marginTop: '1.2rem' }}>
            <h2 className="card-title">Botanical & Decision Boundary Diagnostics</h2>

            <div className="diagnostics-list">
              {insightData.insights?.discriminant_notes?.map((note: string, i: number) => (
                <div key={i} className="diagnostic-item">
                  <span className="diag-icon">🌿</span>
                  <p className="diag-text">{note}</p>
                </div>
              ))}

              {insightData.insights?.kernel_diagnostics?.map((diag: string, i: number) => (
                <div key={i} className="diagnostic-item">
                  <span className="diag-icon">⚙️</span>
                  <p className="diag-text">{diag}</p>
                </div>
              ))}

              {insightData.insights?.outliers && insightData.insights.outliers.length > 0 && (
                <div className="diagnostic-item outlier">
                  <span className="diag-icon">⚠️</span>
                  <p className="diag-text">
                    Outlier warning: {insightData.insights.outliers.join(', ')}
                  </p>
                </div>
              )}
            </div>
          </div>

          {/* Species Alignment & Z-Score Analysis */}
          <div className="card" style={{ marginTop: '1.2rem' }}>
            <h2 className="card-title">Statistical Z-Score & Morphological Distance</h2>
            <p className="text-muted" style={{ marginBottom: '1rem' }}>
              Shows standardized deviation (Z-score) of the sample measurements from each species' canonical distribution mean ($\mu$).
              Values between -2.0 and +2.0 fall within the 95% botanical population confidence interval.
            </p>

            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Target Species</th>
                    <th>Sepal L (Z)</th>
                    <th>Sepal W (Z)</th>
                    <th>Petal L (Z)</th>
                    <th>Petal W (Z)</th>
                    <th>Distance Index</th>
                    <th>Typical Range?</th>
                  </tr>
                </thead>
                <tbody>
                  {insightData.insights?.species_alignment &&
                    Object.entries(insightData.insights.species_alignment).map(([species, info]: [string, any]) => {
                      const isClosest = species === insightData.insights.closest_botanical_match;
                      return (
                        <tr key={species} className={isClosest ? 'best-row' : ''}>
                          <td>
                            <strong style={{ color: SPECIES_COLORS[species] }}>{species}</strong>
                            {isClosest && ' 🏆 (Closest)'}
                          </td>
                          <td><code>{info.z_scores?.sepal_length}</code></td>
                          <td><code>{info.z_scores?.sepal_width}</code></td>
                          <td><code>{info.z_scores?.petal_length}</code></td>
                          <td><code>{info.z_scores?.petal_width}</code></td>
                          <td><strong>{info.morphological_distance}</strong></td>
                          <td>
                            <span className={`status-badge ${info.within_typical_range ? 'active' : 'inactive'}`}>
                              {info.within_typical_range ? 'Yes (Normal)' : 'Atypical (Outlier)'}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Empirical Benchmarks Reference Table */}
          <div className="card" style={{ marginTop: '1.2rem' }}>
            <h2 className="card-title">Empirical Iris Dataset Distribution Benchmarks (N=150)</h2>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Feature</th>
                    <th>Your Sample</th>
                    <th>Setosa Mean ± Std</th>
                    <th>Versicolor Mean ± Std</th>
                    <th>Virginica Mean ± Std</th>
                  </tr>
                </thead>
                <tbody>
                  {['sepal_length', 'sepal_width', 'petal_length', 'petal_width'].map((fKey) => {
                    const stats = insightData.insights?.empirical_benchmarks;
                    const sampleVal = insightData.sample?.[fKey];
                    return (
                      <tr key={fKey}>
                        <td><strong>{FEATURE_NAMES[fKey]}</strong></td>
                        <td><span className="current-sample-pill">{sampleVal} cm</span></td>
                        <td>{stats?.['Iris setosa']?.[fKey]?.mean} ± {stats?.['Iris setosa']?.[fKey]?.std} cm</td>
                        <td>{stats?.['Iris versicolor']?.[fKey]?.mean} ± {stats?.['Iris versicolor']?.[fKey]?.std} cm</td>
                        <td>{stats?.['Iris virginica']?.[fKey]?.mean} ± {stats?.['Iris virginica']?.[fKey]?.std} cm</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {!insightData && !loading && (
        <div className="card empty-state" style={{ marginTop: '1.5rem' }}>
          <div className="empty-icon">💡</div>
          <h3>Generate Morphological Diagnostics</h3>
          <p>
            Choose a preset or adjust the dimensions above, then click <strong>Generate Model Insights</strong> to
            inspect the mathematical decision boundaries and population deviations.
          </p>
        </div>
      )}
    </div>
  );
}
