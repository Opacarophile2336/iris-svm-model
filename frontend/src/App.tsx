import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider } from './auth/AuthContext';
import ProtectedRoute from './routes/ProtectedRoute';
import Layout from './components/Layout';
import Login from './pages/auth/Login';
import Register from './pages/auth/Register';
import Dashboard from './pages/user/Dashboard';
import Datasets from './pages/user/Datasets';
import Quality from './pages/user/Quality';
import Dictionary from './pages/user/Dictionary';
import PredictionStudio from './pages/user/PredictionStudio';
import PredictionHistory from './pages/user/PredictionHistory';
import ModelInsights from './pages/user/ModelInsights';
import ModelHub from './pages/user/ModelHub';
import SpecimenGallery from './pages/user/SpecimenGallery';
import SVMModelLab from './pages/admin/SVMModelLab';
import Evaluation from './pages/admin/Evaluation';
import DecisionBoundary from './pages/admin/DecisionBoundary';
import ModelHistory from './pages/admin/ModelHistory';

export default function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Navigate to="/login" replace />} />
          <Route path="/login" element={<Login />} />
          <Route path="/register" element={<Register />} />

          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >
            <Route path="dashboard" element={<Dashboard />} />
            <Route path="datasets" element={<Datasets />} />
            <Route path="quality" element={<Quality />} />
            <Route path="dictionary" element={<Dictionary />} />
            <Route path="prediction" element={<PredictionStudio />} />
            <Route path="gallery" element={<SpecimenGallery />} />
            <Route path="history" element={<PredictionHistory />} />
            <Route path="insights" element={<ModelInsights />} />
            <Route path="benchmark" element={<ModelHub />} />

            {/* ADMIN-only routes */}
            <Route
              path="admin/svm-lab"
              element={<ProtectedRoute requireAdmin><SVMModelLab /></ProtectedRoute>}
            />
            <Route
              path="admin/evaluation"
              element={<ProtectedRoute requireAdmin><Evaluation /></ProtectedRoute>}
            />
            <Route
              path="admin/decision-boundary"
              element={<ProtectedRoute requireAdmin><DecisionBoundary /></ProtectedRoute>}
            />
            <Route
              path="admin/model-history"
              element={<ProtectedRoute requireAdmin><ModelHistory /></ProtectedRoute>}
            />
          </Route>

          {/* Catch-all */}
          <Route path="*" element={<Navigate to="/login" replace />} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}
