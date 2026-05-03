/**
 * RBAC Authentication Context for HR Attrition Intelligence Dashboard.
 *
 * Manages the full authentication lifecycle:
 * - Login via API key → validate with /v1/auth/whoami
 * - Session persistence via sessionStorage
 * - Role-based permission checking
 * - Logout with session cleanup
 *
 * Backend is the source of truth — this layer is UX only.
 *
 * Roles: admin, hr_partner, analyst, auditor
 * Permissions: predict, override, dashboard, audit, export, system
 */
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { api, type WhoAmI, type LoginResponse, setApiKey, clearApiKey, hasApiKey } from './api';

// ── Permission Constants ─────────────────────────────────────────────
export const Permission = {
  PREDICT: 'predict',
  OVERRIDE: 'override',
  DASHBOARD: 'dashboard',
  AUDIT: 'audit',
  EXPORT: 'export',
  SYSTEM: 'system',
} as const;

export type PermissionValue = (typeof Permission)[keyof typeof Permission];

// ── Role Display Labels ──────────────────────────────────────────────
const ROLE_LABELS: Record<string, string> = {
  admin: 'System Admin',
  hr_partner: 'HR Partner',
  analyst: 'Analyst',
  auditor: 'Auditor',
};

const ROLE_SUBTITLES: Record<string, string> = {
  admin: 'Full Access',
  hr_partner: 'HR Operations',
  analyst: 'Analytics & Reports',
  auditor: 'Compliance Audit',
};

// Role badge gradient colors
export const ROLE_COLORS: Record<string, string> = {
  admin: 'from-emerald-400 to-blue-500',
  hr_partner: 'from-violet-400 to-fuchsia-500',
  analyst: 'from-amber-400 to-orange-500',
  auditor: 'from-slate-400 to-zinc-500',
};

// ── Auth Context ─────────────────────────────────────────────────────
interface AuthContextType {
  /** Whether the user is authenticated */
  isAuthenticated: boolean;
  /** Current user role (e.g., "admin", "analyst") */
  role: string;
  /** List of permissions the current role has */
  permissions: string[];
  /** API key prefix for display purposes */
  apiKeyPrefix: string;
  /** Human-readable role label */
  roleLabel: string;
  /** Role subtitle / department label */
  roleSubtitle: string;
  /** Role badge gradient CSS class */
  roleColor: string;
  /** Check if user has a specific permission */
  can: (permission: PermissionValue) => boolean;
  /** Check if user has ANY of the given permissions */
  canAny: (...permissions: PermissionValue[]) => boolean;
  /** Whether auth data is still loading */
  isLoading: boolean;
  /** Whether auth fetch failed */
  isError: boolean;
  /** Login with username and password. Returns login response on success. */
  login: (username: string, password: string) => Promise<LoginResponse>;
  /** Display name of the current user */
  displayName: string;
  /** Logout and clear session */
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

// ── Dev mode detection ───────────────────────────────────────────────
// In dev mode with no API key configured, skip login requirement
// const ENV_API_KEY = import.meta.env.VITE_API_KEY || '';
const DEV_SKIP_LOGIN = false; // Forced to false to test username/password login

// ── Session Timeout ──────────────────────────────────────────────────
// Sessions expire after 8 hours of inactivity (GDPR best practice).
const SESSION_TIMEOUT_MS = 8 * 60 * 60 * 1000; // 8 hours
const SESSION_CHECK_INTERVAL_MS = 60 * 1000; // Check every 60s
const SESSION_LAST_ACTIVITY_KEY = 'hr_session_last_activity';

function updateLastActivity(): void {
  sessionStorage.setItem(SESSION_LAST_ACTIVITY_KEY, Date.now().toString());
}

function isSessionExpired(): boolean {
  const last = sessionStorage.getItem(SESSION_LAST_ACTIVITY_KEY);
  if (!last) return false; // No timestamp = fresh session, not expired
  return Date.now() - parseInt(last, 10) > SESSION_TIMEOUT_MS;
}

function clearSessionTimestamp(): void {
  sessionStorage.removeItem(SESSION_LAST_ACTIVITY_KEY);
}

// ── AuthProvider ─────────────────────────────────────────────────────
interface AuthProviderProps {
  children: ReactNode;
}

export function AuthProvider({ children }: AuthProviderProps) {
  // Track whether user has authenticated in this session
  const [sessionActive, setSessionActive] = useState(() => {
    // Session is active if there's a key in sessionStorage or env var
    return hasApiKey();
  });

  // Track explicit logout — overrides DEV_SKIP_LOGIN so logout actually works
  const [manuallyLoggedOut, setManuallyLoggedOut] = useState(false);

  const queryClient = useQueryClient();

  // Fetch role info only when session is active
  const { data, isLoading, isError } = useQuery<WhoAmI>({
    queryKey: ['auth-whoami'],
    queryFn: api.getWhoAmI,
    retry: 1,
    staleTime: 60_000,
    refetchInterval: 300_000,
    refetchOnWindowFocus: true,
    enabled: (sessionActive || DEV_SKIP_LOGIN) && !manuallyLoggedOut,
  });

  const login = useCallback(async (username: string, password: string): Promise<LoginResponse> => {
    try {
      // Call login endpoint — returns session token
      const response = await api.login(username, password);
      // Store session token as the API key for subsequent requests
      setApiKey(response.token);
      setSessionActive(true);
      setManuallyLoggedOut(false);
      updateLastActivity();
      // Invalidate all queries to refetch with new credentials
      queryClient.invalidateQueries();
      return response;
    } catch (err) {
      clearApiKey();
      setSessionActive(false);
      throw err;
    }
  }, [queryClient]);

  const logout = useCallback(() => {
    // Invalidate session on server (fire and forget)
    api.logout().catch(() => {});
    clearApiKey();
    clearSessionTimestamp();
    setSessionActive(false);
    setManuallyLoggedOut(true);
    // Clear all cached data from previous session
    queryClient.clear();
  }, [queryClient]);

  // ── Session Timeout Check ────────────────────────────────────────
  // Periodically check if session has expired due to inactivity.
  // Also update last activity on user interactions.
  useEffect(() => {
    if (!sessionActive || DEV_SKIP_LOGIN) return;

    // Check immediately on mount
    if (isSessionExpired()) {
      console.warn('[Auth] Session expired due to inactivity — logging out.');
      logout();
      return;
    }

    // Update activity timestamp on user interaction
    const onActivity = () => updateLastActivity();
    window.addEventListener('click', onActivity);
    window.addEventListener('keydown', onActivity);

    // Periodic check
    const interval = setInterval(() => {
      if (isSessionExpired()) {
        console.warn('[Auth] Session expired due to inactivity — logging out.');
        logout();
      }
    }, SESSION_CHECK_INTERVAL_MS);

    return () => {
      window.removeEventListener('click', onActivity);
      window.removeEventListener('keydown', onActivity);
      clearInterval(interval);
    };
  }, [sessionActive, logout]);

  const value = useMemo<AuthContextType>(() => {
    const isAuthenticated = (sessionActive || DEV_SKIP_LOGIN) && !manuallyLoggedOut;

    // Default permissions when in dev mode without API
    const role = data?.role ?? (DEV_SKIP_LOGIN ? 'admin' : '');
    const permissions = data?.permissions ?? (DEV_SKIP_LOGIN
      ? ['predict', 'override', 'dashboard', 'audit', 'export', 'system']
      : []);
    const apiKeyPrefix = data?.api_key_prefix ?? (DEV_SKIP_LOGIN ? 'dev-mode' : '');

    return {
      isAuthenticated,
      role,
      permissions,
      apiKeyPrefix,
      displayName: data?.display_name || (DEV_SKIP_LOGIN ? 'Admin (Dev)' : ''),
      roleLabel: ROLE_LABELS[role] ?? role,
      roleSubtitle: ROLE_SUBTITLES[role] ?? 'Unknown Role',
      roleColor: ROLE_COLORS[role] ?? 'from-zinc-400 to-zinc-600',
      can: (permission: PermissionValue) => permissions.includes(permission),
      canAny: (...perms: PermissionValue[]) =>
        perms.some((p) => permissions.includes(p)),
      isLoading: isAuthenticated ? isLoading : false,
      isError,
      login,
      logout,
    };
  }, [data, isLoading, isError, sessionActive, manuallyLoggedOut, login, logout]);

  return (
    <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
  );
}

// ── useAuth Hook ─────────────────────────────────────────────────────
export function useAuth(): AuthContextType {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an <AuthProvider>');
  }
  return context;
}

// ── AccessGuard Component ────────────────────────────────────────────
interface AccessGuardProps {
  /** Required permission to render children */
  permission?: PermissionValue;
  /** Alternative: require ANY of these permissions */
  anyPermission?: PermissionValue[];
  /** What to show when access is denied (defaults to nothing) */
  fallback?: ReactNode;
  children: ReactNode;
}

/**
 * Declarative permission guard.
 *
 * @example
 * <AccessGuard permission="predict">
 *   <DecisionCockpit />
 * </AccessGuard>
 *
 * @example
 * <AccessGuard anyPermission={["dashboard", "audit"]} fallback={<AccessDenied />}>
 *   <AnalyticsDashboard />
 * </AccessGuard>
 */
export function AccessGuard({
  permission,
  anyPermission,
  fallback = null,
  children,
}: AccessGuardProps) {
  const { can, canAny, isLoading } = useAuth();

  // While loading, show nothing to prevent flash of unauthorized content
  if (isLoading) return null;

  if (permission && !can(permission)) {
    return <>{fallback}</>;
  }

  if (anyPermission && anyPermission.length > 0 && !canAny(...anyPermission)) {
    return <>{fallback}</>;
  }

  return <>{children}</>;
}
