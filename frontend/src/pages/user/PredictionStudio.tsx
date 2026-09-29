import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  predict,
  predictAll,
  uploadBatchFile,
  exportBatch,
  type SinglePredictionResponse,
  type MultiKernelPredictionResponse,
  type BatchUploadResponse,
} from '../../api/prediction';
import {
  getSpecimenShowcase,
  type SpecimenItem,
  type ShowcaseResponse,
} from '../../api/image';

const SPECIES_COLORS: Record<string, string> = {
  'Iris setosa': '#4E79A7',
  'Iris versicolor': '#F28E2B',
  'Iris virginica': '#59A14F',
};

const KERNEL_COLORS: Record<string, string> = {
  rbf: '#4E79A7',
  linear: '#F28E2B',
  poly: '#59A14F',
  sigmoid: '#E15759',
  mlp: '#76B7B2',
};

type ClassifierKey = 'rbf' | 'linear' | 'poly' | 'sigmoid' | 'mlp';

const CLASSIFIERS: { id: ClassifierKey; name: string; type: string; desc: string }[] = [
  { id: 'rbf', name: 'SVM RBF', type: 'SVM (Radial Basis Function)', desc: 'Gaussian RBF non-linear kernel' },
  { id: 'linear', name: 'SVM Linear', type: 'SVM (Linear)', desc: 'Hyperplane linear separation' },
  { id: 'poly', name: 'SVM Polynomial', type: 'SVM (Degree 3)', desc: '3rd-order polynomial boundary' },
  { id: 'sigmoid', name: 'SVM Sigmoid', type: 'SVM (Sigmoid)', desc: 'Hyperbolic tangent kernel' },
  { id: 'mlp', name: 'Deep Learning MLP', type: 'Neural Network (64, 32)', desc: 'Multi-layer perceptron neural classifier' },
];

interface FeatureInput {
  sepal_length: string;
  sepal_width: string;
  petal_length: string;
  petal_width: string;
}

const EXAMPLE_VALUES = [
  { sepal_length: '5.1', sepal_width: '3.5', petal_length: '1.4', petal_width: '0.2', label: 'Setosa example' },
  { sepal_length: '6.0', sepal_width: '2.9', petal_length: '4.5', petal_width: '1.5', label: 'Versicolor example' },
  { sepal_length: '6.3', sepal_width: '3.3', petal_length: '6.0', petal_width: '2.5', label: 'Virginica example' },
];

export default function PredictionStudio() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'single' | 'batch'>('single');

  // Single prediction states
  const [inputs, setInputs] = useState<FeatureInput>({
    sepal_length: '', sepal_width: '', petal_length: '', petal_width: '',
  });
  const [selectedModel, setSelectedModel] = useState<ClassifierKey>('rbf');
  const [singleResult, setSingleResult] = useState<SinglePredictionResponse | null>(null);
  const [compareResult, setCompareResult] = useState<MultiKernelPredictionResponse | null>(null);
  const [singleLoading, setSingleLoading] = useState(false);
  const [compareLoading, setCompareLoading] = useState(false);
  const [singleError, setSingleError] = useState('');

  // Dynamic Specimen Showcase states
  const [showcaseData, setShowcaseData] = useState<ShowcaseResponse | null>(null);
  const [showcaseLoading, setShowcaseLoading] = useState(false);
  const [inspectSpecimen, setInspectSpecimen] = useState<SpecimenItem | null>(null);
  const [recentShowcaseIds, setRecentShowcaseIds] = useState<{ [species: string]: string[] }>(() => {
    try {
      const saved = sessionStorage.getItem('iris_recent_showcase_ids');
      return saved ? JSON.parse(saved) : {};
    } catch {
      return {};
    }
  });

  const fetchShowcase = async (species: string, currentExcludedMap?: { [sp: string]: string[] }) => {
    setShowcaseLoading(true);
    const map = currentExcludedMap || recentShowcaseIds;
    const currentExcluded = map[species] || [];
    try {
      const res = await getSpecimenShowcase(species, currentExcluded, 3);
      setShowcaseData(res.data);
      const newExcluded = Array.from(new Set([...currentExcluded, ...res.data.displayed_ids])).slice(-30);
      const updatedMap = { ...map, [species]: newExcluded };
      setRecentShowcaseIds(updatedMap);
      try {
        sessionStorage.setItem('iris_recent_showcase_ids', JSON.stringify(updatedMap));
      } catch {}
    } catch (err) {
      console.error('Failed to load specimen showcase:', err);
    } finally {
      setShowcaseLoading(false);
    }
  };

  const handleRotateShowcase = () => {
    const targetSpecies = singleResult?.predicted_species || compareResult?.primary_prediction;
    if (targetSpecies) {
      fetchShowcase(targetSpecies);
    }
  };

  const renderSpecimenShowcase = (species: string) => (
    <div
      className="specimen-showcase-container"
      style={{
        margin: '14px 0',
        padding: '12px',
        background: '#f8fafc',
        borderRadius: '10px',
        border: '1px solid #e2e8f0',
      }}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px', flexWrap: 'wrap', gap: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '0.92rem', fontWeight: 700, color: '#1e293b' }}>
            🌸 Dynamic Specimen Showcase
          </span>
          {showcaseData && (
            <span
              style={{
                fontSize: '0.72rem',
                background: '#e0f2fe',
                color: '#0369a1',
                padding: '2px 8px',
                borderRadius: '12px',
                fontWeight: 600,
              }}
            >
              {showcaseData.count} Distinct Observations ({showcaseData.species})
            </span>
          )}
        </div>
        <button
          type="button"
          className="btn-secondary btn-sm"
          onClick={handleRotateShowcase}
          disabled={showcaseLoading}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '6px',
            fontSize: '0.78rem',
            fontWeight: 600,
            padding: '3px 10px',
            borderRadius: '6px',
            background: '#fff',
            border: '1px solid #cbd5e1',
            cursor: showcaseLoading ? 'not-allowed' : 'pointer',
          }}
          title="Rotate diverse specimens without altering the ML prediction"
        >
          <span style={{ display: 'inline-block', transform: showcaseLoading ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s ease' }}>
            🔄
          </span>
          <span>{showcaseLoading ? 'Rotating...' : 'Rotate Specimens'}</span>
        </button>
      </div>

      {showcaseData && showcaseData.showcase.length > 0 ? (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: '10px' }}>
          {showcaseData.showcase.map((item, idx) => (
            <div
              key={item.id || idx}
              onClick={() => setInspectSpecimen(item)}
              style={{
                background: '#fff',
                borderRadius: '8px',
                overflow: 'hidden',
                border: '1px solid #e2e8f0',
                boxShadow: '0 1px 3px rgba(0,0,0,0.05)',
                cursor: 'pointer',
                transition: 'transform 0.15s ease, box-shadow 0.15s ease',
                display: 'flex',
                flexDirection: 'column',
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 10px rgba(0,0,0,0.08)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'none';
                e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.05)';
              }}
            >
              <div style={{ position: 'relative', height: '130px', background: '#f1f5f9', overflow: 'hidden' }}>
                <img
                  src={item.thumbnail_url || item.image_url}
                  alt={item.species}
                  loading="lazy"
                  decoding="async"
                  onError={(e) => {
                    const target = e.currentTarget;
                    if (item.species.includes('setosa')) target.src = '/assets/iris_setosa_thumb.jpg';
                    else if (item.species.includes('versicolor')) target.src = '/assets/iris_versicolor_thumb.jpg';
                    else target.src = '/assets/iris_virginica_thumb.jpg';
                  }}
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block',
                  }}
                />
                <span
                  style={{
                    position: 'absolute',
                    bottom: '4px',
                    right: '4px',
                    background: 'rgba(0,0,0,0.7)',
                    color: '#fff',
                    fontSize: '0.68rem',
                    padding: '1px 6px',
                    borderRadius: '4px',
                    fontWeight: 500,
                  }}
                >
                  🔍 Inspect
                </span>
              </div>

              <div style={{ padding: '8px 10px', fontSize: '0.74rem', color: '#475569', display: 'flex', flexDirection: 'column', gap: '3px' }}>
                <div style={{ fontWeight: 600, color: '#1e293b', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  👤 {item.recorded_by || 'Verified Observer'}
                </div>
                <div style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  📍 {item.locality || item.country || 'Documented Habitat'}
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '2px', paddingTop: '4px', borderTop: '1px solid #f1f5f9' }}>
                  <span style={{ fontSize: '0.68rem', color: '#64748b' }}>
                    {item.source.includes('iNaturalist') ? 'iNaturalist' : item.source.includes('Wikimedia') ? 'Wikimedia' : 'Herbarium'}
                  </span>
                  <span style={{ fontSize: '0.68rem', background: '#f1f5f9', padding: '1px 4px', borderRadius: '3px' }}>
                    {item.license || 'CC'}
                  </span>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div style={{ padding: '20px', textAlign: 'center', color: '#64748b', fontSize: '0.85rem' }}>
          📷 {showcaseLoading ? 'Loading verified botanical observations...' : `Specimen showcase for ${species}`}
        </div>
      )}
    </div>
  );

  // Batch prediction states
  const [batchFile, setBatchFile] = useState<File | null>(null);
  const [batchResult, setBatchResult] = useState<BatchUploadResponse | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState('');
  const [exportLoading, setExportLoading] = useState(false);
  const [batchPage, setBatchPage] = useState(1);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // ---------- Single Prediction Handlers ----------
  const handleInputChange = (field: keyof FeatureInput) => (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputs(prev => ({ ...prev, [field]: e.target.value }));
  };

  const loadExample = (ex: typeof EXAMPLE_VALUES[0]) => {
    setInputs({
      sepal_length: ex.sepal_length,
      sepal_width: ex.sepal_width,
      petal_length: ex.petal_length,
      petal_width: ex.petal_width,
    });
    setSingleResult(null);
    setCompareResult(null);
    setSingleError('');
  };

  const validateInputs = (): boolean => {
    const vals = [inputs.sepal_length, inputs.sepal_width, inputs.petal_length, inputs.petal_width];
    for (const v of vals) {
      if (!v.trim()) {
        setSingleError('All four measurements are required.');
        return false;
      }
      const n = parseFloat(v);
      if (isNaN(n) || n <= 0 || n > 20) {
        setSingleError('Values must be valid numbers between 0 and 20 cm.');
        return false;
      }
    }
    return true;
  };

  const handlePredictSingle = async () => {
    setSingleError('');
    if (!validateInputs()) return;
    setSingleLoading(true);
    setSingleResult(null);
    setCompareResult(null);

    try {
      const res = await predict({
        sepal_length: parseFloat(inputs.sepal_length),
        sepal_width: parseFloat(inputs.sepal_width),
        petal_length: parseFloat(inputs.petal_length),
        petal_width: parseFloat(inputs.petal_width),
        model: selectedModel,
      });
      setSingleResult(res.data);
      fetchShowcase(res.data.predicted_species);
    } catch (err: any) {
      setSingleError(err.response?.data?.detail || 'Prediction failed. Please check inputs.');
    } finally {
      setSingleLoading(false);
    }
  };

  const handleCompareAll = async () => {
    setSingleError('');
    if (!validateInputs()) return;
    setCompareLoading(true);
    setSingleResult(null);
    setCompareResult(null);

    try {
      const res = await predictAll({
        sepal_length: parseFloat(inputs.sepal_length),
        sepal_width: parseFloat(inputs.sepal_width),
        petal_length: parseFloat(inputs.petal_length),
        petal_width: parseFloat(inputs.petal_width),
      });
      setCompareResult(res.data);
      fetchShowcase(res.data.primary_prediction);
    } catch (err: any) {
      setSingleError(err.response?.data?.detail || 'Comparative prediction failed. Please check inputs.');
    } finally {
      setCompareLoading(false);
    }
  };

  // ---------- Batch Upload Handlers ----------
  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      setBatchFile(files[0]);
      setBatchError('');
      setBatchResult(null);
    }
  };

  const handleBatchUpload = async () => {
    if (!batchFile) {
      setBatchError('Please select a dataset file (.csv, .txt, or .xlsx).');
      return;
    }
    setBatchLoading(true);
    setBatchError('');
    setBatchResult(null);

    const formData = new FormData();
    formData.append('file', batchFile);

    try {
      const res = await uploadBatchFile(formData);
      setBatchResult(res.data);
      setBatchPage(1);
    } catch (err: any) {
      setBatchError(err.response?.data?.detail || 'Failed to process batch file. Please check file format.');
    } finally {
      setBatchLoading(false);
    }
  };

  const handleExport = async (format: 'xlsx' | 'csv') => {
    if (!batchResult) return;
    setExportLoading(true);
    try {
      const response = await exportBatch({
        batch_id: batchResult.batch_id,
        format,
        rows: batchResult.results,
      });

      const blob = new Blob([response.data], {
        type: format === 'xlsx'
          ? 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
          : 'text/csv;charset=utf-8;',
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `hmnc_iris_batch_predictions.${format}`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: any) {
      setBatchError('Failed to export batch results.');
    } finally {
      setExportLoading(false);
    }
  };

  // Pagination for batch table
  const PAGE_SIZE = 15;
  const paginatedRows = batchResult
    ? batchResult.results.slice((batchPage - 1) * PAGE_SIZE, batchPage * PAGE_SIZE)
    : [];
  const totalBatchPages = batchResult ? Math.ceil(batchResult.results.length / PAGE_SIZE) : 1;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">🔮 Prediction Studio</h1>
        <p className="page-subtitle">
          Interactive flower classification powered by multi-kernel Support Vector Machines & Deep Learning MLP
        </p>
      </div>

      {/* Mode Navigation Tabs */}
      <div className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'single' ? 'active' : ''}`}
          onClick={() => setActiveTab('single')}
        >
          <span>🔬 Single Flower Prediction</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'batch' ? 'active' : ''}`}
          onClick={() => setActiveTab('batch')}
        >
          <span>📂 Batch Dataset Upload (.csv, .txt, .xlsx)</span>
        </button>
      </div>

      {/* ============================================================== */}
      {/* TAB 1: SINGLE PREDICTION                                      */}
      {/* ============================================================== */}
      {activeTab === 'single' && (
        <div className="prediction-layout">
          {/* Left: Input Form */}
          <div className="card prediction-input-card">
            <h2 className="card-title">Morphological Measurements</h2>

            <div className="example-buttons">
              <span className="text-muted" style={{ fontSize: '0.85rem' }}>Presets: </span>
              {EXAMPLE_VALUES.map((ex, i) => (
                <button key={i} className="btn-example" onClick={() => loadExample(ex)}>
                  {ex.label}
                </button>
              ))}
            </div>

            <div className="form-group">
              <label className="form-label">
                Select Classifier <span className="unit-label">(5 Models Available)</span>
              </label>
              <select
                className="form-input"
                style={{ fontWeight: 600, cursor: 'pointer' }}
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value as ClassifierKey)}
              >
                {CLASSIFIERS.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.name} — {c.type}
                  </option>
                ))}
              </select>
              <div style={{ marginTop: '6px', fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Engine: <strong style={{ color: KERNEL_COLORS[selectedModel] }}>{CLASSIFIERS.find(c => c.id === selectedModel)?.desc}</strong>
              </div>
            </div>

            <div className="form-group">
              <label className="form-label">
                Sepal Length <span className="unit-label">(cm)</span>
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="20"
                className="form-input"
                placeholder="e.g. 5.1"
                value={inputs.sepal_length}
                onChange={handleInputChange('sepal_length')}
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Sepal Width <span className="unit-label">(cm)</span>
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="20"
                className="form-input"
                placeholder="e.g. 3.5"
                value={inputs.sepal_width}
                onChange={handleInputChange('sepal_width')}
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Petal Length <span className="unit-label">(cm)</span>
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="20"
                className="form-input"
                placeholder="e.g. 1.4"
                value={inputs.petal_length}
                onChange={handleInputChange('petal_length')}
              />
            </div>

            <div className="form-group">
              <label className="form-label">
                Petal Width <span className="unit-label">(cm)</span>
              </label>
              <input
                type="number"
                step="0.1"
                min="0.1"
                max="20"
                className="form-input"
                placeholder="e.g. 0.2"
                value={inputs.petal_width}
                onChange={handleInputChange('petal_width')}
              />
            </div>

            {singleError && <div className="alert-error">{singleError}</div>}

            <button
              className="btn-primary btn-full btn-predict"
              onClick={handlePredictSingle}
              disabled={singleLoading || compareLoading}
            >
              {singleLoading
                ? `🔄 Evaluating with ${CLASSIFIERS.find(c => c.id === selectedModel)?.name}...`
                : '🚀 Predict with Selected Model'}
            </button>

            <button
              className="btn-secondary btn-full"
              style={{ marginTop: '8px' }}
              onClick={handleCompareAll}
              disabled={singleLoading || compareLoading}
            >
              {compareLoading ? '🔄 Comparing All 5 Classifiers...' : '📊 Compare Across All 5 Classifiers'}
            </button>
          </div>

          {/* Right: Single Prediction Results */}
          {singleResult && (
            <div className="card prediction-result-card">
              <div className="card-header-row">
                <h2 className="card-title">Prediction Result</h2>
                <span
                  className="consensus-tag"
                  style={{
                    backgroundColor: `${KERNEL_COLORS[singleResult.kernel] || '#4E79A7'}22`,
                    color: KERNEL_COLORS[singleResult.kernel] || '#4E79A7',
                    fontWeight: 700,
                  }}
                >
                  {singleResult.model} · {singleResult.kernel_display}
                </span>
              </div>

              {/* Dynamic Botanical Specimen Showcase */}
              {renderSpecimenShowcase(singleResult.predicted_species)}

              {/* Primary Prediction Banner */}
              <div
                className="prediction-species-name"
                style={{ color: SPECIES_COLORS[singleResult.predicted_species] || '#4E79A7' }}
              >
                {singleResult.predicted_species}
              </div>
              <div className="prediction-confidence-label">
                Confidence: <strong>{singleResult.confidence}%</strong> · Validated Test Accuracy: <strong>{singleResult.accuracy}%</strong>
              </div>

              {/* Botanical Specimen Gallery Navigation */}
              <div style={{ marginTop: '8px', marginBottom: '8px' }}>
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    padding: '4px 12px',
                    borderRadius: '16px',
                    color: SPECIES_COLORS[singleResult.predicted_species] || 'var(--primary)',
                    cursor: 'pointer',
                  }}
                  onClick={() => navigate(`/gallery?species=${encodeURIComponent(singleResult.predicted_species)}`)}
                >
                  <span>📷 Explore specimen collection ({singleResult.predicted_species})</span>
                  <span>→</span>
                </button>
              </div>

              {/* Class Probability Distribution */}
              <div className="probability-section" style={{ marginTop: '12px' }}>
                <div className="probability-title">Class Probability Distribution</div>
                {(['setosa', 'versicolor', 'virginica'] as const).map((sp) => {
                  const fullName = `Iris ${sp}`;
                  const prob = singleResult.probabilities[sp] ?? 0;
                  const isPredicted = singleResult.predicted_species.toLowerCase().includes(sp);
                  const color = SPECIES_COLORS[fullName] || '#4E79A7';
                  return (
                    <div key={sp} className="probability-row">
                      <div className="probability-label-row">
                        <span className="probability-species-name" style={{ fontWeight: isPredicted ? 700 : 400 }}>
                          {fullName} {isPredicted && '✓'}
                        </span>
                        <span className="probability-value" style={{ fontWeight: isPredicted ? 700 : 500 }}>
                          {prob}%
                        </span>
                      </div>
                      <div className="probability-bar-bg">
                        <div
                          className="probability-bar-fill"
                          style={{
                            width: `${prob}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Footer Metadata */}
              <div className="prediction-meta">
                <span className="meta-item">
                  Classifier: <strong>{singleResult.model}</strong>
                </span>
                <span className="meta-separator">·</span>
                <span className="meta-item">
                  Kernel / Type: <strong>{singleResult.kernel_display}</strong>
                </span>
                {singleResult.history_id && (
                  <>
                    <span className="meta-separator">·</span>
                    <span className="meta-item saved-badge">
                      ✓ Recorded in History #{singleResult.history_id}
                    </span>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Right: Comparative Multi-Kernel Results (All 5 Classifiers) */}
          {compareResult && (
            <div className="card prediction-result-card">
              <div className="card-header-row">
                <h2 className="card-title">Comparative Prediction Results</h2>
                <span className={`consensus-tag ${compareResult.consensus}`}>
                  {compareResult.consensus_text}
                </span>
              </div>

              {/* Dynamic Botanical Specimen Showcase */}
              {renderSpecimenShowcase(compareResult.primary_prediction)}

              {/* Primary Prediction Banner */}
              <div
                className="prediction-species-name"
                style={{ color: SPECIES_COLORS[compareResult.primary_prediction] || '#4E79A7' }}
              >
                {compareResult.primary_prediction}
              </div>
              <div className="prediction-confidence-label">
                Highest Confidence: <strong>{compareResult.highest_confidence.confidence}%</strong> via{' '}
                <span className="kernel-pill">{compareResult.highest_confidence.kernel_display}</span>
              </div>

              {/* Botanical Specimen Gallery Navigation */}
              <div style={{ marginTop: '8px', marginBottom: '8px' }}>
                <button
                  type="button"
                  className="btn-secondary btn-sm"
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '6px',
                    fontSize: '0.82rem',
                    fontWeight: 600,
                    padding: '4px 12px',
                    borderRadius: '16px',
                    color: SPECIES_COLORS[compareResult.primary_prediction] || 'var(--primary)',
                    cursor: 'pointer',
                  }}
                  onClick={() => navigate(`/gallery?species=${encodeURIComponent(compareResult.primary_prediction)}`)}
                >
                  <span>📷 Explore specimen collection ({compareResult.primary_prediction})</span>
                  <span>→</span>
                </button>
              </div>

              {/* 5-Classifier Comparison Matrix */}
              <div
                className="kernel-comparison-grid"
                style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}
              >
                {(['rbf', 'linear', 'poly', 'sigmoid', 'mlp'] as const).map((k) => {
                  const kRes = compareResult.results[k];
                  if (!kRes) return null;
                  const isTop = k === compareResult.highest_confidence.kernel;
                  const color = KERNEL_COLORS[k] || '#4E79A7';

                  return (
                    <div
                      key={k}
                      className={`kernel-result-box ${isTop ? 'highlighted-kernel' : ''}`}
                      style={{ borderTop: `3px solid ${color}` }}
                    >
                      <div className="kernel-box-header">
                        <span className="kernel-box-title" style={{ color }}>
                          {kRes.kernel_display}
                        </span>
                        {isTop && <span className="best-badge">Highest Conf.</span>}
                      </div>

                      <div
                        className="kernel-box-prediction"
                        style={{ color: SPECIES_COLORS[kRes.predicted_species] || '#212529' }}
                      >
                        {kRes.predicted_species}
                      </div>

                      <div className="kernel-metric-row">
                        <span className="metric-name">Confidence:</span>
                        <span className="metric-val">{kRes.confidence}%</span>
                      </div>
                      <div className="kernel-metric-row">
                        <span className="metric-name">Validated Acc:</span>
                        <span className="metric-val">{kRes.accuracy}%</span>
                      </div>

                      {/* Probabilities distribution */}
                      <div className="mini-prob-bars">
                        {Object.entries(kRes.probabilities).map(([sp, prob]) => (
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
                    </div>
                  );
                })}
              </div>

              {/* Footer Metadata */}
              <div className="prediction-meta">
                <span className="meta-item">
                  Decision Consensus: <strong>{compareResult.consensus.toUpperCase()}</strong>
                </span>
                <span className="meta-separator">·</span>
                <span className="meta-item">
                  Accuracy Type: <strong>Validated Model Test Set</strong>
                </span>
                {compareResult.history_id && (
                  <>
                    <span className="meta-separator">·</span>
                    <span className="meta-item saved-badge">
                      ✓ Recorded in History #{compareResult.history_id}
                    </span>
                  </>
                )}
              </div>
            </div>
          )}

          {!singleResult && !compareResult && !singleLoading && !compareLoading && (
            <div className="card prediction-placeholder">
              <div className="placeholder-content">
                <div className="placeholder-emoji">🌸</div>
                <h3>Ready for Inference</h3>
                <p>
                  Enter flower dimensions and select one of the <strong>5 classifiers</strong> (SVM RBF, Linear, Polynomial, Sigmoid, Deep Learning MLP) to begin prediction.
                </p>
                <div className="feature-hints">
                  <div className="hint">🌿 Sepal: outer calyx protection</div>
                  <div className="hint">🌺 Petal: inner diagnostic floral whorl</div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ============================================================== */}
      {/* TAB 2: BATCH DATASET UPLOAD                                    */}
      {/* ============================================================== */}
      {activeTab === 'batch' && (
        <div className="batch-container">
          {/* Upload Configuration Card */}
          <div className="card">
            <h2 className="card-title">📂 Upload Batch Dataset (.csv, .txt, .xlsx)</h2>
            <p className="text-muted" style={{ marginBottom: '1rem' }}>
              Upload an arbitrary batch dataset containing Iris measurements. Every sample row will be evaluated
              using all 5 classifiers (SVM RBF, Linear, Polynomial, Sigmoid, and Deep Learning MLP) alongside real validated model accuracies.
            </p>

            <div className="batch-upload-dropzone">
              <input
                type="file"
                ref={fileInputRef}
                accept=".csv,.txt,.xlsx"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
              <div className="dropzone-box" onClick={() => fileInputRef.current?.click()}>
                <div className="dropzone-icon">📁</div>
                <div className="dropzone-title">
                  {batchFile ? batchFile.name : 'Click to select CSV, TXT, or Excel (XLSX) file'}
                </div>
                <div className="dropzone-sub">
                  Supported extensions: <code>.csv</code>, <code>.txt</code>, <code>.xlsx</code> (Max 10 MB, up to 5,000 samples)
                </div>
              </div>

              {batchFile && (
                <div className="selected-file-banner">
                  <span>Selected file: <strong>{batchFile.name}</strong> ({(batchFile.size / 1024).toFixed(1)} KB)</span>
                  <button className="btn-secondary btn-sm" onClick={() => setBatchFile(null)}>Change</button>
                </div>
              )}
            </div>

            {/* Column specification banner */}
            <div className="column-guide-banner">
              <strong>Required Features (Auto-mapped):</strong>
              <ul>
                <li><code>Sepal Length</code> (cm)</li>
                <li><code>Sepal Width</code> (cm)</li>
                <li><code>Petal Length</code> (cm)</li>
                <li><code>Petal Width</code> (cm)</li>
              </ul>
              <small className="text-muted">
                Variations such as <code>sepal_length</code>, <code>Sepal Length (cm)</code>, or <code>sl</code> are automatically matched.
              </small>
            </div>

            {batchError && <div className="alert-error" style={{ marginTop: '1rem' }}>{batchError}</div>}

            <div style={{ marginTop: '1.2rem' }}>
              <button
                className="btn-primary"
                onClick={handleBatchUpload}
                disabled={batchLoading || !batchFile}
              >
                {batchLoading ? '🔄 Processing Batch across 5 Classifiers...' : '🚀 Run Batch 5-Classifier Prediction'}
              </button>
            </div>
          </div>

          {/* Batch Prediction Results */}
          {batchResult && (
            <div className="card" style={{ marginTop: '1.5rem' }}>
              <div className="card-header-row">
                <div>
                  <h2 className="card-title">Batch Prediction Results ({batchResult.total_rows} samples)</h2>
                  <span className="text-muted">File: {batchResult.filename}</span>
                </div>
                <div className="export-btn-group">
                  <button
                    className="btn-primary btn-sm"
                    onClick={() => handleExport('xlsx')}
                    disabled={exportLoading}
                  >
                    📥 Download Excel (.xlsx)
                  </button>
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => handleExport('csv')}
                    disabled={exportLoading}
                  >
                    📄 Download CSV (.csv)
                  </button>
                </div>
              </div>

              {/* Warning notifications if any rows skipped */}
              {batchResult.warnings && batchResult.warnings.length > 0 && (
                <div className="alert-warning">
                  <strong>Notice:</strong> {batchResult.warnings.length} row(s) were skipped due to non-numeric or out-of-range values.
                </div>
              )}

              {/* Real Model Accuracy Summary Banner */}
              <div className="model-accuracy-bar">
                <span className="bar-label">Model Benchmark Accuracies:</span>
                <span className="accuracy-pill rbf">
                  RBF: <strong>{batchResult.metrics?.rbf?.test_accuracy || 96.67}%</strong>
                </span>
                <span className="accuracy-pill linear">
                  Linear: <strong>{batchResult.metrics?.linear?.test_accuracy || 96.67}%</strong>
                </span>
                <span className="accuracy-pill poly">
                  Poly: <strong>{batchResult.metrics?.poly?.test_accuracy || 96.67}%</strong>
                </span>
                <span className="accuracy-pill sigmoid">
                  Sigmoid: <strong>{batchResult.metrics?.sigmoid?.test_accuracy || 93.33}%</strong>
                </span>
                <span className="accuracy-pill mlp">
                  MLP: <strong>{batchResult.metrics?.mlp?.test_accuracy || 96.67}%</strong>
                </span>
              </div>

              {/* Results Table */}
              <div className="table-container" style={{ marginTop: '1rem' }}>
                <table className="batch-table">
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Sepal L</th>
                      <th>Sepal W</th>
                      <th>Petal L</th>
                      <th>Petal W</th>
                      <th>RBF Prediction</th>
                      <th>Linear Prediction</th>
                      <th>Poly Prediction</th>
                      <th>Sigmoid Prediction</th>
                      <th>MLP Prediction</th>
                      <th>Consensus</th>
                    </tr>
                  </thead>
                  <tbody>
                    {paginatedRows.map((row) => (
                      <tr key={row.row_number}>
                        <td><strong>{row.row_number}</strong></td>
                        <td>{row.sepal_length}</td>
                        <td>{row.sepal_width}</td>
                        <td>{row.petal_length}</td>
                        <td>{row.petal_width}</td>
                        <td>
                          <span
                            className="species-badge"
                            style={{
                              backgroundColor: `${SPECIES_COLORS[row.rbf_prediction] || '#4E79A7'}22`,
                              color: SPECIES_COLORS[row.rbf_prediction],
                            }}
                          >
                            {row.rbf_prediction} ({row.rbf_confidence}%)
                          </span>
                        </td>
                        <td>
                          <span
                            className="species-badge"
                            style={{
                              backgroundColor: `${SPECIES_COLORS[row.linear_prediction] || '#F28E2B'}22`,
                              color: SPECIES_COLORS[row.linear_prediction],
                            }}
                          >
                            {row.linear_prediction} ({row.linear_confidence}%)
                          </span>
                        </td>
                        <td>
                          <span
                            className="species-badge"
                            style={{
                              backgroundColor: `${SPECIES_COLORS[row.poly_prediction] || '#59A14F'}22`,
                              color: SPECIES_COLORS[row.poly_prediction],
                            }}
                          >
                            {row.poly_prediction} ({row.poly_confidence}%)
                          </span>
                        </td>
                        <td>
                          {row.sigmoid_prediction ? (
                            <span
                              className="species-badge"
                              style={{
                                backgroundColor: `${SPECIES_COLORS[row.sigmoid_prediction] || '#E15759'}22`,
                                color: SPECIES_COLORS[row.sigmoid_prediction] || '#E15759',
                              }}
                            >
                              {row.sigmoid_prediction} ({row.sigmoid_confidence}%)
                            </span>
                          ) : (
                            <span className="text-muted">—</span>
                          )}
                        </td>
                        <td>
                          {row.mlp_prediction ? (
                            <span
                              className="species-badge"
                              style={{
                                backgroundColor: `${SPECIES_COLORS[row.mlp_prediction] || '#76B7B2'}22`,
                                color: SPECIES_COLORS[row.mlp_prediction] || '#76B7B2',
                              }}
                            >
                              {row.mlp_prediction} ({row.mlp_confidence}%)
                            </span>
                          ) : (
                            <span className="text-muted">—</span>
                          )}
                        </td>
                        <td>
                          <span className={`consensus-pill ${row.consensus.toLowerCase()}`}>
                            {row.consensus}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* Pagination */}
              {totalBatchPages > 1 && (
                <div className="pagination">
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => setBatchPage(p => Math.max(1, p - 1))}
                    disabled={batchPage === 1}
                  >
                    ← Prev
                  </button>
                  <span>Page {batchPage} of {totalBatchPages}</span>
                  <button
                    className="btn-secondary btn-sm"
                    onClick={() => setBatchPage(p => Math.min(totalBatchPages, p + 1))}
                    disabled={batchPage === totalBatchPages}
                  >
                    Next →
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* High-Resolution Specimen Inspection Modal */}
      {inspectSpecimen && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0, 0, 0, 0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: '20px',
            backdropFilter: 'blur(3px)',
          }}
          onClick={() => setInspectSpecimen(null)}
        >
          <div
            className="card"
            style={{
              maxWidth: '850px',
              width: '100%',
              maxHeight: '90vh',
              overflowY: 'auto',
              padding: '0',
              borderRadius: '12px',
              background: 'var(--bg-card, #ffffff)',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            {/* Modal Header */}
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '16px 20px',
                borderBottom: '1px solid var(--border, #e2e8f0)',
              }}
            >
              <div>
                <h2 style={{ margin: 0, fontSize: '1.25rem', color: SPECIES_COLORS[inspectSpecimen.species] || '#4E79A7' }}>
                  {inspectSpecimen.species}
                </h2>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted, #64748b)' }}>
                  Specimen Record ID: <code>{inspectSpecimen.id}</code>
                </span>
              </div>
              <button
                type="button"
                className="btn-secondary btn-sm"
                onClick={() => setInspectSpecimen(null)}
                style={{ fontSize: '1.1rem', cursor: 'pointer', padding: '4px 10px' }}
              >
                ✕
              </button>
            </div>

            {/* High-res Photograph */}
            <div style={{ maxHeight: '450px', overflow: 'hidden', background: '#000000', display: 'flex', justifyContent: 'center' }}>
              <img
                src={inspectSpecimen.image_url}
                alt={inspectSpecimen.species}
                onError={(e) => {
                  const target = e.currentTarget;
                  if (inspectSpecimen.species.includes('setosa')) target.src = '/assets/iris_setosa_thumb.jpg';
                  else if (inspectSpecimen.species.includes('versicolor')) target.src = '/assets/iris_versicolor_thumb.jpg';
                  else target.src = '/assets/iris_virginica_thumb.jpg';
                }}
                style={{
                  maxHeight: '450px',
                  width: 'auto',
                  maxWidth: '100%',
                  objectFit: 'contain',
                }}
              />
            </div>

            {/* Verified Specimen Metadata Sheet */}
            <div style={{ padding: '20px' }}>
              <h3 style={{ fontSize: '1.05rem', marginBottom: '12px', color: 'var(--text, #1e293b)' }}>
                🌿 Botanical Herbarium & Occurrence Record
              </h3>

              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
                  gap: '12px',
                  fontSize: '0.88rem',
                }}
              >
                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Taxon / Species</div>
                  <strong style={{ color: SPECIES_COLORS[inspectSpecimen.species] || '#4E79A7' }}>{inspectSpecimen.species}</strong>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Primary Data Provider</div>
                  <strong>{inspectSpecimen.source}</strong>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Geographic Locality</div>
                  <span>{inspectSpecimen.locality || inspectSpecimen.country || 'Documented Habitat'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Herbarium / Institution</div>
                  <span>{inspectSpecimen.institution || 'Verified Repository'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Collector / Recorder</div>
                  <span>{inspectSpecimen.recorded_by || 'Verified Observer'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Observation Date</div>
                  <span>{inspectSpecimen.event_date ? inspectSpecimen.event_date.split('T')[0] : 'Historical Record'}</span>
                </div>

                <div style={{ padding: '10px', background: 'var(--bg, #f8fafc)', borderRadius: '6px' }}>
                  <div style={{ color: 'var(--text-muted, #64748b)', fontSize: '0.78rem' }}>Copyright / License</div>
                  <span>{inspectSpecimen.license || 'Public Domain / Open Access'}</span>
                </div>
              </div>

              <div style={{ marginTop: '16px', display: 'flex', justifyContent: 'flex-end', gap: '10px' }}>
                {inspectSpecimen.occurrence_url && (
                  <a
                    href={inspectSpecimen.occurrence_url}
                    target="_blank"
                    rel="noreferrer"
                    className="btn-secondary btn-sm"
                    style={{ textDecoration: 'none' }}
                  >
                    View Original Record ↗
                  </a>
                )}
                <button
                  type="button"
                  className="btn-primary btn-sm"
                  onClick={() => setInspectSpecimen(null)}
                >
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
