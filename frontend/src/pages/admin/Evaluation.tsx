import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { getEvaluation } from '../../api/ml';

const CLASSIFIER_KEYS = ['rbf', 'linear', 'poly', 'sigmoid', 'mlp'];

const CLASSIFIER_LABELS: Record<string, string> = {
  rbf: 'SVM RBF',
  linear: 'SVM Linear',
  poly: 'SVM Polynomial',
  sigmoid: 'SVM Sigmoid',
  mlp: 'Deep Learning MLP',
};

const CLASSIFIER_TYPES: Record<string, string> = {
  rbf: 'Support Vector Machine (RBF Kernel)',
  linear: 'Support Vector Machine (Linear Kernel)',
  poly: 'Support Vector Machine (Polynomial Kernel degree=3)',
  sigmoid: 'Support Vector Machine (Sigmoid Kernel)',
  mlp: 'Artificial Neural Network (4 → 64 → 32 → 3 Softmax)',
};

const CLASSIFIER_COLORS: Record<string, string> = {
  rbf: '#4E79A7',
  linear: '#F28E2B',
  poly: '#59A14F',
  sigmoid: '#7970A1',
  mlp: '#9C27B0',
};

const CLASS_NAMES = ['Iris setosa', 'Iris versicolor', 'Iris virginica'];

export default function Evaluation() {
  const [evalData, setEvalData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getEvaluation()
      .then(res => setEvalData(res.data))
      .catch(() => setError('Không thể tải dữ liệu đánh giá mô hình.'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page">
        <div className="loading-spinner">Đang tải và tính toán dữ liệu đánh giá 5 mô hình...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="page">
        <div className="alert-error">{error}</div>
      </div>
    );
  }

  // Active classifiers returned by backend
  const availableClassifiers = CLASSIFIER_KEYS.filter(k => evalData && evalData[k]);

  // Comparative chart data
  const metricsChartData = [
    {
      metric: 'Test Accuracy',
      ...Object.fromEntries(
        availableClassifiers.map(k => [
          CLASSIFIER_LABELS[k],
          Math.round((evalData?.[k]?.test_accuracy || 0) * 1000) / 10,
        ])
      ),
    },
    {
      metric: '5-Fold CV Mean',
      ...Object.fromEntries(
        availableClassifiers.map(k => [
          CLASSIFIER_LABELS[k],
          Math.round((evalData?.[k]?.cv_mean || 0) * 1000) / 10,
        ])
      ),
    },
    {
      metric: 'Precision (Weighted)',
      ...Object.fromEntries(
        availableClassifiers.map(k => [
          CLASSIFIER_LABELS[k],
          Math.round((evalData?.[k]?.precision || 0) * 1000) / 10,
        ])
      ),
    },
    {
      metric: 'Recall (Weighted)',
      ...Object.fromEntries(
        availableClassifiers.map(k => [
          CLASSIFIER_LABELS[k],
          Math.round((evalData?.[k]?.recall || 0) * 1000) / 10,
        ])
      ),
    },
    {
      metric: 'F1-Score (Weighted)',
      ...Object.fromEntries(
        availableClassifiers.map(k => [
          CLASSIFIER_LABELS[k],
          Math.round((evalData?.[k]?.f1 || 0) * 1000) / 10,
        ])
      ),
    },
  ];

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">📈 Đánh giá hiệu năng mô hình (Model Evaluation)</h1>
        <p className="page-subtitle">
          Báo cáo kiểm thử thực tế và ma trận nhầm lẫn (Confusion Matrix) độc lập cho toàn bộ 5 bộ phân loại: SVM RBF, SVM Linear, SVM Polynomial, SVM Sigmoid, và Deep Learning MLP.
        </p>
      </div>

      {/* Comparative Performance Chart */}
      <div className="card" style={{ marginBottom: '24px' }}>
        <h2 className="card-title">📊 So sánh đa chỉ số giữa 5 bộ phân loại (Benchmark Comparison)</h2>
        <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem', marginBottom: '16px' }}>
          Tất cả các giá trị phần trăm (%) được tính toán trực tiếp trên tập kiểm thử độc lập (Test Set 20%) và kiểm định chéo 5 phần (5-Fold Cross-Validation).
        </p>
        <div style={{ width: '100%', height: 340 }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={metricsChartData} margin={{ top: 20, right: 30, left: 10, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
              <XAxis dataKey="metric" tick={{ fontSize: 12 }} />
              <YAxis domain={[80, 100]} unit="%" tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value: any) => [`${value}%`]} />
              <Legend wrapperStyle={{ paddingTop: '10px' }} />
              {availableClassifiers.map(k => (
                <Bar
                  key={k}
                  dataKey={CLASSIFIER_LABELS[k]}
                  fill={CLASSIFIER_COLORS[k]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Individual Classifier Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {availableClassifiers.map(k => {
          const d = evalData[k];
          const color = CLASSIFIER_COLORS[k] || '#4E79A7';
          const label = CLASSIFIER_LABELS[k] || k.toUpperCase();
          const typeDesc = CLASSIFIER_TYPES[k] || '';

          return (
            <div
              key={k}
              className="card"
              style={{
                borderLeft: `5px solid ${color}`,
                padding: '24px',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
                <div>
                  <h2 className="card-title" style={{ color: color, fontSize: '1.25rem', marginBottom: '2px' }}>
                    {label}
                  </h2>
                  <span style={{ fontSize: '0.82rem', color: 'var(--text-muted)' }}>{typeDesc}</span>
                </div>
                <span
                  style={{
                    backgroundColor: `${color}18`,
                    color: color,
                    border: `1px solid ${color}44`,
                    padding: '4px 10px',
                    borderRadius: '4px',
                    fontWeight: 600,
                    fontSize: '0.8rem',
                  }}
                >
                  Test Accuracy: {(d.test_accuracy * 100).toFixed(2)}%
                </span>
              </div>

              {/* Metrics Pill Grid */}
              <div className="metrics-grid" style={{ marginBottom: '18px' }}>
                <div className="metric-pill" style={{ backgroundColor: `${color}11`, borderColor: `${color}44` }}>
                  <div className="metric-label">Độ chính xác kiểm thử (Test Accuracy)</div>
                  <div className="metric-value" style={{ color: color }}>{(d.test_accuracy * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-pill">
                  <div className="metric-label">Độ chuẩn xác (Precision)</div>
                  <div className="metric-value">{(d.precision * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-pill">
                  <div className="metric-label">Độ bao quát (Recall)</div>
                  <div className="metric-value">{(d.recall * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-pill">
                  <div className="metric-label">Điểm F1 (F1-Score)</div>
                  <div className="metric-value">{(d.f1 * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-pill">
                  <div className="metric-label">Độ chính xác CV trung bình (5-Fold CV)</div>
                  <div className="metric-value">{(d.cv_mean * 100).toFixed(2)}%</div>
                </div>
                <div className="metric-pill">
                  <div className="metric-label">Độ lệch chuẩn CV (CV Std)</div>
                  <div className="metric-value">±{(d.cv_std * 100).toFixed(2)}%</div>
                </div>
              </div>

              {/* Confusion Matrix */}
              {d.confusion_matrix && (
                <div style={{ marginTop: '12px', borderTop: '1px solid var(--border-light)', paddingTop: '16px' }}>
                  <h3 className="card-subtitle" style={{ marginBottom: '10px' }}>
                    Ma trận nhầm lẫn (Confusion Matrix — 30 mẫu kiểm thử)
                  </h3>
                  <div className="confusion-matrix-container">
                    <div className="cm-label-row">
                      <div className="cm-corner" style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Thực tế \ Dự đoán</div>
                      {CLASS_NAMES.map(c => (
                        <div key={c} className="cm-header" style={{ fontSize: '0.78rem' }}>
                          {c.replace('Iris ', '')}
                        </div>
                      ))}
                    </div>
                    {d.confusion_matrix.map((row: number[], i: number) => (
                      <div key={i} className="cm-row">
                        <div className="cm-row-label" style={{ fontSize: '0.78rem', fontWeight: 600 }}>
                          {CLASS_NAMES[i].replace('Iris ', '')}
                        </div>
                        {row.map((cell: number, j: number) => (
                          <div
                            key={j}
                            className={`cm-cell ${i === j ? 'cm-diagonal' : cell > 0 ? 'cm-off-diag' : ''}`}
                            title={`Thực tế: ${CLASS_NAMES[i]} — Dự đoán: ${CLASS_NAMES[j]} (${cell} mẫu)`}
                          >
                            <strong>{cell}</strong>
                          </div>
                        ))}
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
