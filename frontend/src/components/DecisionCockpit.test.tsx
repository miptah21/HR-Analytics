/**
 * Integration tests for DecisionCockpit component.
 *
 * Uses MSW to mock API responses at the network level,
 * and a fresh QueryClient per test for isolation.
 */
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import DecisionCockpit from '../components/DecisionCockpit';
import { server } from '../test/server';
import { http, HttpResponse } from 'msw';

// ── MSW Lifecycle ────────────────────────────────────────────────────
beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }));
afterEach(() => server.resetHandlers());
afterAll(() => server.close());

// ── Test Utility ─────────────────────────────────────────────────────
function renderWithProviders(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      {ui}
    </QueryClientProvider>
  );
}

// ── Tests ────────────────────────────────────────────────────────────
describe('DecisionCockpit', () => {
  it('renders simulation controls', () => {
    renderWithProviders(<DecisionCockpit />);
    expect(screen.getByText('Simulation Controls')).toBeInTheDocument();
    expect(screen.getByText(/Salary Hike/)).toBeInTheDocument();
    expect(screen.getByText(/Monthly Income/)).toBeInTheDocument();
  });

  it('shows risk probability after API response', async () => {
    renderWithProviders(<DecisionCockpit />);
    // MSW returns 0.42 probability → 42.0%
    await waitFor(
      () => {
        expect(screen.getByText(/42\.0%/)).toBeInTheDocument();
      },
      { timeout: 5000 }
    );
  });

  it('shows risk tier badge', async () => {
    renderWithProviders(<DecisionCockpit />);
    await waitFor(
      () => {
        expect(screen.getByText(/Medium Risk/)).toBeInTheDocument();
      },
      { timeout: 5000 }
    );
  });

  it('displays SHAP explainability disclaimer', async () => {
    renderWithProviders(<DecisionCockpit />);
    await waitFor(
      () => {
        expect(
          screen.getByText(/statistical correlations/i)
        ).toBeInTheDocument();
      },
      { timeout: 5000 }
    );
  });

  it('displays the override button', async () => {
    renderWithProviders(<DecisionCockpit />);
    await waitFor(
      () => {
        expect(screen.getByText('Override AI')).toBeInTheDocument();
      },
      { timeout: 5000 }
    );
  });

  it('shows override form when Override AI is clicked', async () => {
    const user = userEvent.setup();
    renderWithProviders(<DecisionCockpit />);

    // Wait for prediction to load
    await waitFor(() => {
      expect(screen.getByText(/42\.0%/)).toBeInTheDocument();
    }, { timeout: 5000 });

    // Click Override AI button
    const overrideBtn = screen.getByText('Override AI');
    await user.click(overrideBtn);

    // Override form should appear
    expect(screen.getByPlaceholderText(/Explain why you disagree/)).toBeInTheDocument();
    expect(screen.getByText('Submit Override')).toBeInTheDocument();
    expect(screen.getByText('Cancel')).toBeInTheDocument();
  });

  it('handles API error gracefully', async () => {
    server.use(
      http.post('http://localhost:8000/v1/predict', () => {
        return HttpResponse.json(
          { detail: 'Service temporarily unavailable' },
          { status: 503 }
        );
      })
    );

    renderWithProviders(<DecisionCockpit />);
    await waitFor(
      () => {
        expect(screen.getByText(/API unavailable/)).toBeInTheDocument();
      },
      { timeout: 5000 }
    );
  });

  it('overtime toggle switches between Yes and No', async () => {
    const user = userEvent.setup();
    renderWithProviders(<DecisionCockpit />);

    const noBtn = screen.getByText('No');
    await user.click(noBtn);

    // The "No" button should now be active (emerald styling)
    expect(noBtn.className).toContain('emerald');
  });
});
