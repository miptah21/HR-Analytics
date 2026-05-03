/**
 * Admin Dashboard — User Management Panel
 *
 * Full CRUD interface for managing system users:
 * - View all users in a premium data table
 * - Create new users with role assignment
 * - Edit user details, role, and active status
 * - Delete users (with confirmation)
 * - Reset passwords
 *
 * Only accessible to admin role (system permission).
 */
import { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Users, UserPlus, Pencil, Trash2, Shield, ShieldCheck,
  ShieldAlert, ShieldOff, Check, Eye, EyeOff,
  AlertCircle, Loader2, Search,
} from 'lucide-react';
import { api, type UserRecord, type CreateUserData, type UpdateUserData } from '../lib/api';

// Role config for display
const ROLE_CONFIG: Record<string, { label: string; color: string; icon: typeof Shield }> = {
  admin: { label: 'Admin', color: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20', icon: ShieldCheck },
  hr_partner: { label: 'HR Partner', color: 'text-violet-400 bg-violet-500/10 border-violet-500/20', icon: Shield },
  analyst: { label: 'Analyst', color: 'text-amber-400 bg-amber-500/10 border-amber-500/20', icon: ShieldAlert },
  auditor: { label: 'Auditor', color: 'text-zinc-400 bg-zinc-500/10 border-zinc-500/20', icon: ShieldOff },
};

function RoleBadge({ role }: { role: string }) {
  const config = ROLE_CONFIG[role] || ROLE_CONFIG.analyst;
  const Icon = config.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border ${config.color}`}>
      <Icon className="w-3 h-3" />
      {config.label}
    </span>
  );
}

function StatusBadge({ active }: { active: boolean }) {
  return active ? (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium text-green-400 bg-green-500/10 border border-green-500/20">
      <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
      Active
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs font-medium text-red-400 bg-red-500/10 border border-red-500/20">
      <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
      Disabled
    </span>
  );
}

// ── Create User Modal ────────────────────────────────────────────────
function CreateUserModal({ onClose }: { onClose: () => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<CreateUserData>({
    username: '', password: '', display_name: '', role: 'analyst',
  });
  const [showPwd, setShowPwd] = useState(false);
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: () => api.createUser(form),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
      onClose();
    },
    onError: (err: unknown) => {
      setError(
        typeof err === 'object' && err !== null && 'detail' in err
          ? String((err as { detail: string }).detail)
          : 'Failed to create user.'
      );
    },
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-md bg-white border border-slate-200 rounded-2xl p-6 shadow-tonal"
      >
        <h3 className="text-lg font-semibold text-slate-900 flex items-center gap-2 mb-5">
          <UserPlus className="w-5 h-5 text-blue-400" />
          Create New User
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Username</label>
            <input
              value={form.username}
              onChange={(e) => setForm({ ...form, username: e.target.value })}
              placeholder="e.g. john.doe"
              className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-slate-900 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Display Name</label>
            <input
              value={form.display_name}
              onChange={(e) => setForm({ ...form, display_name: e.target.value })}
              placeholder="e.g. John Doe"
              className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-slate-900 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Password</label>
            <div className="relative">
              <input
                type={showPwd ? 'text' : 'password'}
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                placeholder="Min. 6 characters"
                className="w-full px-3 py-2.5 pr-10 bg-white border border-slate-200 rounded-xl text-slate-900 text-sm placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-teal-500/40"
              />
              <button type="button" onClick={() => setShowPwd(!showPwd)} className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400">
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-500 mb-1.5">Role</label>
            <select
              value={form.role}
              onChange={(e) => setForm({ ...form, role: e.target.value })}
              className="w-full px-3 py-2.5 bg-white border border-slate-200 rounded-xl text-slate-900 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/40 appearance-none"
            >
              <option value="admin">Admin</option>
              <option value="hr_partner">HR Partner</option>
              <option value="analyst">Analyst</option>
              <option value="auditor">Auditor</option>
            </select>
          </div>

          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-xl">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
              <span className="text-red-300 text-sm">{error}</span>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button onClick={onClose} className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
            Cancel
          </button>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending || !form.username || !form.password}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2"
          >
            {mutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
            Create User
          </button>
        </div>
      </motion.div>
    </div>
  );
}

// ── Edit User Modal ──────────────────────────────────────────────────
function EditUserModal({ user, onClose }: { user: UserRecord; onClose: () => void }) {
  const queryClient = useQueryClient();
  const [form, setForm] = useState<UpdateUserData>({
    display_name: user.display_name || '',
    role: user.role,
    is_active: user.is_active,
  });
  const [newPassword, setNewPassword] = useState('');
  const [showPwd, setShowPwd] = useState(false);
  const [error, setError] = useState('');

  const mutation = useMutation({
    mutationFn: () => {
      const data: UpdateUserData = { ...form };
      if (newPassword) data.password = newPassword;
      return api.updateUser(user.id, data);
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
      onClose();
    },
    onError: (err: unknown) => {
      setError(typeof err === 'object' && err !== null && 'detail' in err
        ? String((err as { detail: string }).detail) : 'Failed to update user.');
    },
  });

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.95 }}
        className="w-full max-w-md bg-zinc-900 border border-white/[0.08] rounded-2xl p-6 shadow-2xl"
      >
        <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-5">
          <Pencil className="w-5 h-5 text-amber-400" />
          Edit User: {user.username}
        </h3>

        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">Display Name</label>
            <input
              value={form.display_name || ''}
              onChange={(e) => setForm({ ...form, display_name: e.target.value })}
              className="w-full px-3 py-2.5 bg-white/[0.04] border border-white/[0.08] rounded-xl text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40"
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">Role</label>
            <select
              value={form.role || user.role}
              onChange={(e) => setForm({ ...form, role: e.target.value })}
              className="w-full px-3 py-2.5 bg-white/[0.04] border border-white/[0.08] rounded-xl text-white text-sm focus:outline-none focus:ring-2 focus:ring-blue-500/40 appearance-none"
            >
              <option value="admin" className="bg-zinc-900">Admin</option>
              <option value="hr_partner" className="bg-zinc-900">HR Partner</option>
              <option value="analyst" className="bg-zinc-900">Analyst</option>
              <option value="auditor" className="bg-zinc-900">Auditor</option>
            </select>
          </div>

          <div className="flex items-center justify-between">
            <label className="text-xs font-medium text-zinc-400">Active Status</label>
            <button
              onClick={() => setForm({ ...form, is_active: !form.is_active })}
              className={`relative w-11 h-6 rounded-full transition-colors ${form.is_active ? 'bg-green-600' : 'bg-zinc-700'}`}
            >
              <span className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform ${form.is_active ? 'translate-x-5' : ''}`} />
            </button>
          </div>

          <div>
            <label className="block text-xs font-medium text-zinc-400 mb-1.5">New Password (leave blank to keep current)</label>
            <div className="relative">
              <input
                type={showPwd ? 'text' : 'password'}
                value={newPassword}
                onChange={(e) => setNewPassword(e.target.value)}
                placeholder="Enter new password..."
                className="w-full px-3 py-2.5 pr-10 bg-white/[0.04] border border-white/[0.08] rounded-xl text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
              />
              <button type="button" onClick={() => setShowPwd(!showPwd)} className="absolute right-3 top-1/2 -translate-y-1/2 text-zinc-500">
                {showPwd ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {error && (
            <div className="flex items-center gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-xl">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
              <span className="text-red-300 text-sm">{error}</span>
            </div>
          )}
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button onClick={onClose} className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
            Cancel
          </button>
          <button
            onClick={() => mutation.mutate()}
            disabled={mutation.isPending}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 disabled:bg-zinc-700 text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2"
          >
            {mutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Check className="w-4 h-4" />}
            Save Changes
          </button>
        </div>
      </motion.div>
    </div>
  );
}

// ── Main Admin Dashboard ─────────────────────────────────────────────
export default function AdminDashboard() {
  const queryClient = useQueryClient();
  const [showCreate, setShowCreate] = useState(false);
  const [editUser, setEditUser] = useState<UserRecord | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<UserRecord | null>(null);
  const [search, setSearch] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['admin-users'],
    queryFn: api.getUsers,
    staleTime: 30_000,
  });

  const deleteMutation = useMutation({
    mutationFn: (userId: number) => api.deleteUser(userId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['admin-users'] });
      setDeleteConfirm(null);
    },
  });

  const users = data?.users || [];
  const filtered = users.filter(u =>
    u.username.toLowerCase().includes(search.toLowerCase()) ||
    (u.display_name || '').toLowerCase().includes(search.toLowerCase()) ||
    u.role.toLowerCase().includes(search.toLowerCase())
  );

  // Stats
  const totalUsers = users.length;
  const activeUsers = users.filter(u => u.is_active).length;
  const roleBreakdown = users.reduce<Record<string, number>>((acc, u) => {
    acc[u.role] = (acc[u.role] || 0) + 1;
    return acc;
  }, {});

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 text-blue-400 animate-spin" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-red-400">
        <AlertCircle className="w-8 h-8 mx-auto mb-2" />
        Failed to load users. Check your permissions.
      </div>
    );
  }

  return (
    <div className="space-y-6 p-1">
      {/* Stats Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-tonal">
          <p className="text-xs text-slate-500 uppercase tracking-wider">Total Users</p>
          <p className="text-2xl font-bold text-slate-900 mt-1">{totalUsers}</p>
        </div>
        <div className="bg-white border border-slate-200 rounded-xl p-4 shadow-tonal">
          <p className="text-xs text-slate-500 uppercase tracking-wider">Active</p>
          <p className="text-2xl font-bold text-emerald-500 mt-1">{activeUsers}</p>
        </div>
        {Object.entries(roleBreakdown).map(([role, count]) => (
          <div key={role} className="bg-white border border-slate-200 rounded-xl p-4 shadow-tonal">
            <p className="text-xs text-slate-500 uppercase tracking-wider">{ROLE_CONFIG[role]?.label || role}</p>
            <p className="text-2xl font-bold text-slate-900 mt-1">{count}</p>
          </div>
        ))}
      </div>

      {/* Toolbar */}
      <div className="flex items-center justify-between gap-4">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search users..."
            className="w-full pl-10 pr-4 py-2.5 bg-white/[0.04] border border-white/[0.08] rounded-xl text-white text-sm placeholder:text-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500/40"
          />
        </div>
        <button
          onClick={() => setShowCreate(true)}
          className="px-4 py-2.5 bg-blue-600 hover:bg-blue-500 text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2 shrink-0"
        >
          <UserPlus className="w-4 h-4" />
          Add User
        </button>
      </div>

      {/* Users Table */}
      <div className="bg-white border border-slate-200 shadow-tonal rounded-xl overflow-hidden">
        <table className="w-full">
          <thead>
            <tr className="border-b border-slate-200 bg-slate-50">
              <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-4 py-3">User</th>
              <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-4 py-3">Role</th>
              <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-4 py-3">Status</th>
              <th className="text-left text-xs font-medium text-zinc-500 uppercase tracking-wider px-4 py-3">Last Login</th>
              <th className="text-right text-xs font-medium text-zinc-500 uppercase tracking-wider px-4 py-3">Actions</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((user, i) => (
              <motion.tr
                key={user.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.03 }}
                className="border-b border-slate-100 hover:bg-slate-50 transition-colors"
              >
                <td className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-blue-500/20 to-violet-500/20 border border-blue-500/10 flex items-center justify-center">
                      <span className="text-sm font-bold text-blue-600">
                        {(user.display_name || user.username).charAt(0).toUpperCase()}
                      </span>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-slate-900">{user.display_name || user.username}</p>
                      <p className="text-xs text-slate-500">@{user.username}</p>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3"><RoleBadge role={user.role} /></td>
                <td className="px-4 py-3"><StatusBadge active={user.is_active} /></td>
                <td className="px-4 py-3">
                  <span className="text-xs text-zinc-500">
                    {user.last_login
                      ? new Date(user.last_login).toLocaleDateString('id-ID', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })
                      : 'Never'}
                  </span>
                </td>
                <td className="px-4 py-3">
                  <div className="flex items-center justify-end gap-1">
                    <button
                      onClick={() => setEditUser(user)}
                      className="p-2 text-zinc-500 hover:text-blue-400 hover:bg-blue-500/10 rounded-lg transition-colors"
                      title="Edit"
                    >
                      <Pencil className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => setDeleteConfirm(user)}
                      className="p-2 text-zinc-500 hover:text-red-400 hover:bg-red-500/10 rounded-lg transition-colors"
                      title="Delete"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </td>
              </motion.tr>
            ))}
          </tbody>
        </table>

        {filtered.length === 0 && (
          <div className="py-12 text-center text-zinc-500">
            <Users className="w-8 h-8 mx-auto mb-2 opacity-50" />
            {search ? 'No users match your search.' : 'No users found.'}
          </div>
        )}
      </div>

      {/* Modals */}
      <AnimatePresence>
        {showCreate && <CreateUserModal onClose={() => setShowCreate(false)} />}
        {editUser && <EditUserModal user={editUser} onClose={() => setEditUser(null)} />}
      </AnimatePresence>

      {/* Delete Confirmation */}
      <AnimatePresence>
        {deleteConfirm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              className="w-full max-w-sm bg-zinc-900 border border-red-500/20 rounded-2xl p-6 shadow-2xl"
            >
              <h3 className="text-lg font-semibold text-white flex items-center gap-2">
                <Trash2 className="w-5 h-5 text-red-400" />
                Delete User
              </h3>
              <p className="mt-3 text-sm text-zinc-400">
                Are you sure you want to delete <strong className="text-white">{deleteConfirm.username}</strong>?
                This action cannot be undone.
              </p>
              <div className="flex justify-end gap-3 mt-6">
                <button onClick={() => setDeleteConfirm(null)} className="px-4 py-2 text-sm text-zinc-400 hover:text-white transition-colors">
                  Cancel
                </button>
                <button
                  onClick={() => deleteMutation.mutate(deleteConfirm.id)}
                  disabled={deleteMutation.isPending}
                  className="px-4 py-2 bg-red-600 hover:bg-red-500 disabled:bg-zinc-700 text-white text-sm font-medium rounded-xl transition-colors flex items-center gap-2"
                >
                  {deleteMutation.isPending ? <Loader2 className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
                  Delete
                </button>
              </div>
            </motion.div>
          </div>
        )}
      </AnimatePresence>
    </div>
  );
}
