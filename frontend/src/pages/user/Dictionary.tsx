import { useEffect, useState } from 'react';
import { getDictionary } from '../../api/datasets';

const TYPE_COLORS: Record<string, string> = {
  continuous: '#4E79A7',
  categorical: '#F28E2B',
};

export default function Dictionary() {
  const [dictionary, setDictionary] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getDictionary().then(res => setDictionary(res.data)).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="page"><div className="loading-spinner">Loading dictionary...</div></div>;

  return (
    <div className="page">
      <div className="page-header">
        <h1 className="page-title">Feature Dictionary</h1>
        <p className="page-subtitle">Definitions, units, and descriptions for all dataset features</p>
      </div>

      <div className="card">
        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Display Name</th>
                <th>Unit</th>
                <th>Description</th>
                <th>Range</th>
                <th>Type</th>
              </tr>
            </thead>
            <tbody>
              {dictionary.map((item) => (
                <tr key={item.feature}>
                  <td><code>{item.feature}</code></td>
                  <td><strong>{item.display_name}</strong></td>
                  <td>{item.unit || '—'}</td>
                  <td>{item.description}</td>
                  <td>{item.range}</td>
                  <td>
                    <span className="type-badge" style={{
                      backgroundColor: `${TYPE_COLORS[item.type]}22`,
                      color: TYPE_COLORS[item.type],
                      border: `1px solid ${TYPE_COLORS[item.type]}55`,
                    }}>
                      {item.type}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="card" style={{ marginTop: '1rem' }}>
        <h2 className="card-title">Target Classes</h2>
        <div className="class-list">
          {['Iris setosa', 'Iris versicolor', 'Iris virginica'].map((cls, i) => (
            <div key={cls} className="class-item">
              <span className="class-number">{i}</span>
              <span className="class-name">{cls}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
