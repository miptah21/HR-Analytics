/**
 * Centralized API client for the HR Attrition Intelligence API.
 *
 * Handles base URL configuration, authentication headers, and
 * error response normalization.
 *
 * API key is sourced from sessionStorage (set by login flow)
 * with a fallback to VITE_API_KEY for backwards compatibility.
 */

export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// ── Session-Based API Key Management ─────────────────────────────────
const SESSION_KEY = 'hr_api_key';

/** Store API key in sessionStorage (cleared when browser tab closes). */
export function setApiKey(key: string): void {
  sessionStorage.setItem(SESSION_KEY, key);
}

/** Retrieve the active API key. Session key takes priority over env var. */
export function getApiKey(): string {
  return sessionStorage.getItem(SESSION_KEY) || import.meta.env.VITE_API_KEY || '';
}

/** Clear stored API key (logout). */
export function clearApiKey(): void {
  sessionStorage.removeItem(SESSION_KEY);
}

/** Check if an API key is configured (session or env). */
export function hasApiKey(): boolean {
  return getApiKey().length > 0;
}

// ── Request Helper ───────────────────────────────────────────────────
interface ApiError {
  status: number;
  detail: string;
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {},
): Promise<T> {
  const apiKey = getApiKey();
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(apiKey ? { 'X-API-Key': apiKey } : {}),
    ...(options.headers as Record<string, string> || {}),
  };

  const res = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (!res.ok) {
    let detail = `API Error: ${res.status}`;
    try {
      const body = await res.json();
      if (Array.isArray(body.detail)) {
        detail = body.detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ');
      } else if (typeof body.detail === 'object' && body.detail !== null) {
        detail = JSON.stringify(body.detail);
      } else {
        detail = body.detail || detail;
      }
    } catch {
      // Response body is not JSON
    }
    const error: ApiError = { status: res.status, detail };
    throw error;
  }

  return res.json();
}

// ── Typed API Methods ────────────────────────────────────────────────

export interface PredictionRequest {
  EmployeeID: string;
  Age: number;
  JobRole: string;
  JobLevel: number;
  MonthlyIncome: number;
  PercentSalaryHike: number;
  OverTime: string;
  DistanceFromHome: number;
  WorkLifeBalance: number;
  YearsAtCompany: number;
  YearsInCurrentRole: number;
  YearsSinceLastPromotion: number;
  YearsWithCurrManager: number;
  TotalWorkingYears: number;
  JobSatisfaction: number;
  EnvironmentSatisfaction: number;
  RelationshipSatisfaction: number;
  JobInvolvement: number;
  BusinessTravel: string;
}

export interface PredictionResponse {
  EmployeeID: string;
  Risk_Probability: number;
  Risk_Tier: 'High' | 'Medium' | 'Low';
  Expected_Financial_Loss: number;
  Top_Risk_Drivers: Record<string, number>;
  Retention_Strategy: string;
  Recommended_Action: string;
  Causal_Uplift_Score?: number;
  Uplift_Recommendation?: string;
  Explainability_Disclaimer: string;
  Causal_Warning: string;
  Model_Version: string;
}

export interface OverrideRequest {
  employee_id: string;
  prediction_log_id?: number;
  original_risk_tier: string;
  override_risk_tier: string;
  override_reason: string;
}

export interface InterventionLogRequest {
  employee_id: string;
  prediction_log_id?: number;
  intervention_type: string;
  intervention_details?: string;
}

export interface OverrideResponse {
  id: number;
  message: string;
}

export interface DashboardSummary {
  total_employees_scored: number;
  high_risk_count: number;
  medium_risk_count: number;
  low_risk_count: number;
  total_value_at_risk: number;
  average_risk_probability: number;
  top_systemic_drivers: Array<Record<string, unknown>>;
}

export interface EmployeeRiskScore {
  EmployeeNumber: number;
  Department: string;
  JobRole: string;
  Predicted_Probability: number;
  Actual: number;
  Risk_Tier: 'High' | 'Medium' | 'Low';
  MonthlyIncome: number;
  Annual_Salary: number;
  Replacement_Cost: number;
  Expected_Loss: number;
  PerformanceRating?: number;
  JobSatisfaction?: number;
  EnvironmentSatisfaction?: number;
  RelationshipSatisfaction?: number;
  WorkLifeBalance?: number;
  YearsAtCompany?: number;
  PercentSalaryHike?: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  stats_loaded: boolean;
  version: string;
}

export interface DriftFeature {
  feature: string;
  baseline_importance: number;
  current_importance: number;
  relative_change: number;
  direction: 'increased' | 'decreased';
}

export interface SHAPDriftReport {
  analysis_date: string;
  verdict: 'STABLE' | 'DRIFT_DETECTED' | 'baseline_established';
  has_drift: boolean;
  metrics: {
    spearman_rank_correlation: number;
    spearman_p_value: number;
    rank_drift_detected: boolean;
    threshold: number;
    top5_overlap: number;
    top5_baseline: string[];
    top5_current: string[];
    top5_stable: boolean;
    features_with_magnitude_drift: number;
  };
  drifted_features: DriftFeature[];
  recommendation: string;
  status?: string;
  message?: string;
}

export interface DriftStatus {
  data_drift: Record<string, unknown> | null;
  shap_drift: SHAPDriftReport | null;
}

export interface TrendDataPoint {
  date: string;
  risk_tier: string;
  count: number;
  avg_probability: number;
}

export interface TrendResponse {
  period_days: number;
  data: TrendDataPoint[];
}

export interface CohortData {
  cohort: string;
  count: number;
  avg_probability: number;
  high_risk_count: number;
  total_value_at_risk: number;
}

export interface CohortResponse {
  group_by: string;
  cohorts: CohortData[];
}

export interface WhoAmI {
  role: string;
  permissions: string[];
  api_key_prefix: string;
  display_name?: string;
}

export interface LoginResponse {
  token: string;
  role: string;
  permissions: string[];
  display_name: string;
  expires_in_hours: number;
}

export interface UserRecord {
  id: number;
  username: string;
  display_name: string | null;
  role: string;
  is_active: boolean;
  created_at: string | null;
  updated_at: string | null;
  last_login: string | null;
}

export interface CreateUserData {
  username: string;
  password: string;
  display_name?: string;
  role: string;
}

export interface UpdateUserData {
  display_name?: string;
  role?: string;
  is_active?: boolean;
  password?: string;
}

export const api = {
  // ── Auth ──────────────────────────────────────────────────────
  login: (username: string, password: string) =>
    request<LoginResponse>('/v1/auth/login', {
      method: 'POST',
      body: JSON.stringify({ username, password }),
    }),

  logout: () =>
    request<{ message: string }>('/v1/auth/logout', { method: 'POST' }),

  getWhoAmI: () =>
    request<WhoAmI>('/v1/auth/whoami'),

  // ── Predict / Override ────────────────────────────────────────
  predict: (data: PredictionRequest) =>
    request<PredictionResponse>('/v1/predict', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  override: (data: OverrideRequest) =>
    request<OverrideResponse>('/v1/override', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  logIntervention: (data: InterventionLogRequest) =>
    request<OverrideResponse>('/v1/intervention', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // ── Dashboard ─────────────────────────────────────────────────
  getDashboardSummary: () =>
    request<DashboardSummary>('/v1/dashboard/summary'),

  getDashboardEmployees: () =>
    request<EmployeeRiskScore[]>('/v1/dashboard/employees'),

  getEmployeeNarrative: (employeeId: number) =>
    request<any>(`/v1/dashboard/employees/${employeeId}/narrative`),

  getEmployeeProfile: (employeeId: number) =>
    request<PredictionRequest>(`/v1/dashboard/employees/${employeeId}/profile`),

  getDriftStatus: () =>
    request<DriftStatus>('/v1/dashboard/drift'),

  getTrends: (days: number = 30) =>
    request<TrendResponse>(`/v1/dashboard/trends?days=${days}`),

  getCohorts: (groupBy: string = 'Department') =>
    request<CohortResponse>(`/v1/dashboard/cohorts?group_by=${groupBy}`),

  exportData: (format: string = 'json') =>
    request<{ export_date: string; total_records: number; data: Record<string, unknown>[] }>(
      `/v1/reports/export?format=${format}`
    ),

  // ── Admin: User Management ────────────────────────────────────
  getUsers: () =>
    request<{ total: number; users: UserRecord[] }>('/v1/admin/users'),

  createUser: (data: CreateUserData) =>
    request<{ message: string; user: UserRecord }>('/v1/admin/users', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  updateUser: (userId: number, data: UpdateUserData) =>
    request<{ message: string; user: UserRecord }>(`/v1/admin/users/${userId}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    }),

  deleteUser: (userId: number) =>
    request<{ message: string }>(`/v1/admin/users/${userId}`, {
      method: 'DELETE',
    }),

  // ── System ────────────────────────────────────────────────────
  healthCheck: () =>
    request<HealthResponse>('/v1/health'),
};

