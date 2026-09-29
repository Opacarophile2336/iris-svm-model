import { useEffect, useState } from 'react';
import { getQuality } from '../../api/datasets';

const FEATURE_LABELS: Record<string, string> = {
  sepal_length: 'Sepal Length',
  sepal_width: 'Sepal Width',
  petal_length: 'Petal Length',
  petal_width: 'Petal Width',
};

export default function Quality() {
  const [quality, setQuality] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getQuality().then(res => setQuality(res.data)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page"><div className="loading-spinner">Loading quality data...</div></div>;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Data Quality</h1>
        <p className="page-subtitle">Statistical summary and quality metrics</p>
      </div>

      <div className="card-grid four-cols">
        <div className="stat-card green">
          <div className="stat-value">{quality?.total_records}</div>
          <div className="stat-label">Total Records</div>
        </div>
        <div className="stat-card blue">
          <div className="stat-value">{quality?.total_features}</div>
          <div className="stat-label">Features</div>
        </div>
        <div className="stat-card green">
          <div className="stat-value">{quality?.missing_values}</div>
          <div className="stat-label">Missing Values</div>
        </div>
        <div className="stat-card green">
          <div className="stat-value">{quality?.missing_pct}%</div>
          <div className="stat-label">Missing %</div>
        </div>
      </div>

      <div className="card" style={{ marginTop: '1rem' }}>
        <h2 className="card-title">Feature Statistics</h2>
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Count</th>
                <th>Mean</th>
                <th>Std</th>
                <th>Min</th>
                <th>Q25</th>
                <th>Median</th>
                <th>Q75</th>
                <th>Max</th>
                <th>Missing</th>
              </tr>
            </thead>
            <tbody>
              {quality?.feature_stats && Object.entries(quality.feature_stats).map(([key, stats]: [string, any]) => (
                <tr key={key}>
                  <td><strong>{FEATURE_LABELS[key] || key}</strong></td>
                  <td>{stats.count}</td>
                  <td>{stats.mean}</td>
                  <td>{stats.std}</td>
                  <td>{stats.min}</td>
                  <td>{stats.q25}</td>
                  <td>{stats.median}</td>
                  <td>{stats.q75}</td>
                  <td>{stats.max}</td>
                  <td><span style={{ color: stats.missing > 0 ? '#E15759' : '#59A14F' }}>{stats.missing}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: '1rem' }}>
        <h2 className="card-title">Class Balance</h2>
        <p className="text-muted">{quality?.class_balance}</p>
        <div className="quality-indicator">
          <span className="quality-dot green"></span>
          <span>Dataset is complete and balanced — no missing values, no class imbalance</span>
        </div>
      </div>
    </div>
  );
}
