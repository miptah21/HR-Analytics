/**
 * MSW (Mock Service Worker) handlers for testing.
 * Intercepts API calls at the network level so components
 * exercise their real fetch logic.
 */
import { http, HttpResponse } from 'msw';

const API_BASE = 'http://localhost:8000';

export const mockPredictionResponse = {
  EmployeeID: 'EMP-042',
  Risk_Probability: 0.42,
  Risk_Tier: 'Medium' as const,
  Expected_Financial_Loss: 55440.0,
  Top_Risk_Drivers: {
    over_time_yes: 0.312,
    years_since_last_promotion: 0.187,
    work_life_balance: 0.143,
  },
  Retention_Strategy: 'Consider work-life balance improvements and promotion pathway.',
  Recommended_Action: 'Remove Overtime (Lowers risk by 8.2%)',
  Explainability_Disclaimer:
    'SHAP values reflect statistical correlations in historical data, not causal relationships.',
  Model_Version: 'abc123def456',
};

export const mockOverrideResponse = {
  id: 1,
  message: 'Override recorded successfully.',
};

export const mockHealthResponse = {
  status: 'healthy',
  model_loaded: true,
  stats_loaded: true,
  version: '1.0.0',
};

export const handlers = [
  // POST /v1/predict
  http.post(`${API_BASE}/v1/predict`, async () => {
    return HttpResponse.json(mockPredictionResponse);
  }),

  // POST /v1/override
  http.post(`${API_BASE}/v1/override`, async () => {
    return HttpResponse.json(mockOverrideResponse);
  }),

  // GET /v1/health
  http.get(`${API_BASE}/v1/health`, () => {
    return HttpResponse.json(mockHealthResponse);
  }),
];
