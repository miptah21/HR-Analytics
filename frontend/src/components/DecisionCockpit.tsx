import { useState, useCallback, useEffect, useRef } from 'react';
import { useMutation } from '@tanstack/react-query';
import { Card, Tracker } from '@tremor/react';
import { motion } from 'framer-motion';
import { Sparkles, TrendingDown, AlertTriangle, Activity, Lightbulb, ThumbsDown } from 'lucide-react';
import { api, type PredictionRequest, type PredictionResponse, type OverrideRequest } from '../lib/api';

const INITIAL_EMPLOYEE: PredictionRequest = {
  EmployeeID: "EMP-042",
  Age: 32,
  JobRole: "Sales Executive",
  JobLevel: 2,
  MonthlyIncome: 5500,
  PercentSalaryHike: 12,
  OverTime: "Yes",
  DistanceFromHome: 15,
  WorkLifeBalance: 2,
  YearsAtCompany: 5,
  YearsInCurrentRole: 4,
  YearsSinceLastPromotion: 4,
  YearsWithCurrManager: 4,
  TotalWorkingYears: 8,
  JobSatisfaction: 2,
  EnvironmentSatisfaction: 2,
  RelationshipSatisfaction: 3,
  JobInvolvement: 3,
  BusinessTravel: "Travel_Rarely"
};

export default function DecisionCockpit() {
  const [formData, setFormData] = useState(INITIAL_EMPLOYEE);
  const debounceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [showOverrideForm, setShowOverrideForm] = useState(false);
  const [overrideReason, setOverrideReason] = useState('');
  
  const predictionMutation = useMutation({
    mutationFn: (data: PredictionRequest) => api.predict(data),
    onError: (error: any) => {
      console.error('Prediction failed:', error?.detail || error);
    },
  });

  const overrideMutation = useMutation({
    mutationFn: (data: OverrideRequest) => api.override(data),
    onSuccess: () => {
      setShowOverrideForm(false);
      setOverrideReason('');
    },
  });

  const interventionMutation = useMutation({
    mutationFn: (data: { employee_id: string; intervention_type: string }) => api.logIntervention(data),
  });

  // Debounced prediction trigger
  const triggerPrediction = useCallback((newData: PredictionRequest) => {
    if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
    debounceTimerRef.current = setTimeout(() => {
      predictionMutation.mutate(newData);
    }, 800); // Increased debounce to 800ms to allow LLM processing without spamming
  }, [predictionMutation]);

  const handleSliderChange = (key: keyof PredictionRequest, value: number) => {
    const newData = { ...formData, [key]: value };
    setFormData(newData);
    triggerPrediction(newData);
  };

  const handleToggle = (key: keyof PredictionRequest, value: string) => {
    const newData = { ...formData, [key]: value };
    setFormData(newData);
    triggerPrediction(newData);
  };

  const handleOverrideSubmit = () => {
    if (!data || overrideReason.length < 10) return;
    overrideMutation.mutate({
      employee_id: formData.EmployeeID,
      original_risk_tier: data.Risk_Tier,
      override_risk_tier: data.Risk_Tier === 'High' ? 'Low' : 'High',
      override_reason: overrideReason,
    });
  };

  // Trigger initial prediction on first render
  useEffect(() => {
    const simulatorEmployeeId = localStorage.getItem('simulatorEmployeeId');
    if (simulatorEmployeeId) {
      localStorage.removeItem('simulatorEmployeeId');
      api.getEmployeeProfile(parseInt(simulatorEmployeeId))
        .then((profileData) => {
          setFormData(profileData as PredictionRequest);
          predictionMutation.mutate(profileData as PredictionRequest);
        })
        .catch((err) => {
          console.error("Failed to fetch employee profile:", err);
          predictionMutation.mutate(formData);
        });
    } else {
      predictionMutation.mutate(formData);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const data: PredictionResponse | undefined = predictionMutation.data;
  const prob = data?.Risk_Probability ? (data.Risk_Probability * 100).toFixed(1) : '0';
  
  const renderDrivers = () => {
    if (!data?.Top_Risk_Drivers) return null;
    return Object.entries(data.Top_Risk_Drivers).map(([key, val], i) => (
      <div key={i} className="flex items-start gap-3 p-3 bg-red-50 border border-red-200 rounded-lg mb-2">
        <AlertTriangle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
        <p className="text-sm text-slate-600">
          <span className="font-semibold text-slate-900">{key.replace(/_/g, ' ')}</span> is correlated with higher flight risk (+{(val as number).toFixed(2)} impact).
        </p>
      </div>
    ));
  };

  return (
    <motion.div 
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="p-8 max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-12 gap-8"
    >
      {/* LEFT PANEL: What-If Simulator */}
      <div className="xl:col-span-5 space-y-6">
        <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
          <div className="mb-6">
            <h3 className="text-xl font-medium text-slate-900">Employee State Profile</h3>
            <p className="text-sm text-slate-500">Manual what-if simulation is disabled to prevent causal misinterpretation. Rely on the DiCE Counterfactual Engine for valid interventions.</p>
          </div>

          {/* API Error */}
          {predictionMutation.isError && (
            <div className="mb-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
              <p className="text-sm text-red-300">
                ⚠ API unavailable — {(predictionMutation.error as any)?.detail || 'check that the backend is running'}
              </p>
            </div>
          )}

          <div className="space-y-5">
            {/* Salary Hike Display */}
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-600">Salary Hike (%)</label>
                <span className="text-sm font-bold text-slate-900">{formData.PercentSalaryHike}%</span>
              </div>
              <input 
                id="slider-salary-hike"
                type="range" min="10" max="25" step="1" 
                value={formData.PercentSalaryHike} 
                disabled
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-not-allowed opacity-50"
              />
            </div>

            {/* Monthly Income Display */}
            <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-600">Monthly Income ($)</label>
                <span className="text-sm font-bold text-slate-900">${Math.round(formData.MonthlyIncome).toLocaleString()}</span>
              </div>
              <input 
                id="slider-monthly-income"
                type="range" min="2000" max="20000" step="500" 
                value={formData.MonthlyIncome} 
                disabled
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-not-allowed opacity-50"
              />
            </div>

             {/* Work Life Balance Display */}
             <div>
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium text-slate-600">Work-Life Balance (1-4)</label>
                <span className="text-sm font-bold text-slate-900">{formData.WorkLifeBalance}</span>
              </div>
              <input 
                id="slider-work-life-balance"
                type="range" min="1" max="4" step="1" 
                value={formData.WorkLifeBalance} 
                disabled
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-not-allowed opacity-50"
              />
            </div>

            {/* Overtime Display */}
            <div className="flex items-center justify-between pt-2">
              <label className="text-sm font-medium text-slate-600">Requires Overtime?</label>
              <div className="flex gap-2">
                <button 
                  disabled
                  className={`px-4 py-1.5 text-xs font-medium rounded-md opacity-70 cursor-not-allowed ${formData.OverTime === 'Yes' ? 'bg-red-50 text-red-600 border border-red-200' : 'bg-slate-100 text-slate-500'}`}
                >
                  Yes
                </button>
                <button 
                  disabled
                  className={`px-4 py-1.5 text-xs font-medium rounded-md opacity-70 cursor-not-allowed ${formData.OverTime === 'No' ? 'bg-emerald-50 text-emerald-600 border border-emerald-200' : 'bg-slate-100 text-slate-500'}`}
                >
                  No
                </button>
              </div>
            </div>
            
            {/* Job Satisfaction Display */}
             <div>
              <div className="flex justify-between mb-1 pt-2">
                <label className="text-sm font-medium text-slate-600">Job Satisfaction (1-4)</label>
                <span className="text-sm font-bold text-slate-900">{formData.JobSatisfaction}</span>
              </div>
              <input 
                id="slider-job-satisfaction"
                type="range" min="1" max="4" step="1" 
                value={formData.JobSatisfaction} 
                disabled
                className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-not-allowed opacity-50"
              />
            </div>
          </div>
        </Card>
      </div>

      {/* RIGHT PANEL: Live Results */}
      <div className="xl:col-span-7 space-y-6">
        {/* Risk Gauge Header */}
        <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-8 relative overflow-hidden">
          <div className="absolute top-0 right-0 p-8 opacity-5">
            <Activity className="w-32 h-32 text-slate-900" />
          </div>
          
          <p className="text-sm font-medium text-slate-500 mb-2">Live Probability of Default (Flight Risk)</p>
          <div className="flex items-end gap-4 mb-4">
            <h2 className={`text-6xl font-bold tracking-tighter ${Number(prob) > 50 ? 'text-error' : 'text-emerald-500'}`}>
              {predictionMutation.isPending ? '...' : `${prob}%`}
            </h2>
            <span className={`px-3 py-1 mb-2 rounded-full text-xs font-bold ${
              data?.Risk_Tier === 'High' ? 'bg-error/10 text-error' : 
              data?.Risk_Tier === 'Medium' ? 'bg-amber-100 text-amber-600' : 
              'bg-emerald-100 text-emerald-600'
            }`}>
              {data?.Risk_Tier || 'Loading'} Risk
            </span>
          </div>

          <p className="text-slate-600 font-medium">Expected Financial Impact:</p>
          <p className="text-2xl text-slate-900 font-bold mb-6">
            ${data?.Expected_Financial_Loss ? Math.round(data.Expected_Financial_Loss).toLocaleString() : '0'}
          </p>

          <Tracker 
            data={Array(20).fill(0).map((_, i) => ({
              color: i < (Number(prob) / 5) ? (Number(prob) > 50 ? "rose" : "emerald") : "zinc"
            }))} 
            className="mt-2 w-full h-4 mb-6"
          />

          {/* Counterfactual Smart Action Recommender (DiCE) */}
          {data?.Recommended_Action && (
            <motion.div 
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className="mt-4 p-3 bg-blue-50 border border-blue-200 rounded-lg flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <Lightbulb className="w-5 h-5 text-blue-600" />
                <p className="text-sm font-medium text-blue-900">
                  <span className="text-blue-700 font-bold">DiCE Causal Intervention: </span> 
                  {data.Recommended_Action}
                </p>
              </div>
            </motion.div>
          )}

          {/* Causal Uplift Assessment (T-Learner) & Intervention Logging */}
          {data?.Uplift_Recommendation && (
            <motion.div 
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              className={`mt-4 p-4 border rounded-lg flex flex-col gap-3 ${data.Causal_Uplift_Score && data.Causal_Uplift_Score < -5 ? 'bg-emerald-50 border-emerald-200' : 'bg-slate-50 border-slate-200'}`}
            >
              <div className="flex items-start gap-3">
                <Activity className={`w-5 h-5 mt-0.5 ${data.Causal_Uplift_Score && data.Causal_Uplift_Score < -5 ? 'text-emerald-600' : 'text-slate-400'}`} />
                <div>
                  <p className={`text-sm font-bold ${data.Causal_Uplift_Score && data.Causal_Uplift_Score < -5 ? 'text-emerald-800' : 'text-slate-700'}`}>
                    Causal ROI Assessment (T-Learner)
                  </p>
                  <p className={`text-sm mt-1 ${data.Causal_Uplift_Score && data.Causal_Uplift_Score < -5 ? 'text-emerald-700' : 'text-slate-600'}`}>
                    {data.Uplift_Recommendation}
                  </p>
                </div>
              </div>
              
              {data.Causal_Uplift_Score && data.Causal_Uplift_Score < -5 && (
                <div className="flex justify-end pt-2 border-t border-emerald-200/50">
                  <button 
                    onClick={() => interventionMutation.mutate({ 
                      employee_id: formData.EmployeeID, 
                      intervention_type: "Salary Hike >15%" 
                    })}
                    disabled={interventionMutation.isPending || interventionMutation.isSuccess}
                    className="px-4 py-1.5 text-xs font-bold bg-emerald-600 hover:bg-emerald-700 text-white rounded shadow-sm transition-colors disabled:opacity-50"
                  >
                    {interventionMutation.isPending ? 'Logging...' : interventionMutation.isSuccess ? '✓ Intervention Logged' : 'Log as Applied Intervention'}
                  </button>
                </div>
              )}
            </motion.div>
          )}
        </Card>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Explainability (The "Why") */}
          <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6">
             <div className="flex items-center gap-2 mb-4">
              <TrendingDown className="w-5 h-5 text-slate-400" />
              <h3 className="text-lg font-medium text-slate-900">Top Risk Drivers</h3>
             </div>
             <div className="space-y-1">
               {predictionMutation.isPending ? (
                 <div className="animate-pulse flex space-x-4">
                   <div className="flex-1 space-y-4 py-1">
                     <div className="h-10 bg-slate-100 rounded"></div>
                     <div className="h-10 bg-slate-100 rounded"></div>
                   </div>
                 </div>
               ) : renderDrivers()}
             </div>
             {/* SHAP Disclaimer (GAP-20) */}
             {data && (
               <>
                 <p className="mt-3 text-xs text-slate-500 italic">
                   {data.Explainability_Disclaimer}
                 </p>
                 {data.Causal_Warning && (
                   <div className="mt-2 p-2 bg-amber-50 border border-amber-200 rounded-lg">
                     <p className="text-xs text-amber-600">
                       ⚠ {data.Causal_Warning}
                     </p>
                   </div>
                 )}
               </>
             )}
          </Card>

          {/* Gemini AI Copilot */}
          <Card className="bg-white border-slate-200 shadow-tonal ring-0 rounded-2xl p-6 relative">
            <div className="absolute inset-0 bg-gradient-to-br from-teal-50 to-transparent rounded-2xl pointer-events-none" />
             <div className="flex items-center gap-2 mb-4">
              <Sparkles className="w-5 h-5 text-teal-dark" />
              <h3 className="text-lg font-medium text-teal-dark">AI Strategy Copilot</h3>
             </div>
             
             {predictionMutation.isPending ? (
                 <div className="animate-pulse space-y-2">
                   <div className="h-4 bg-slate-100 rounded w-3/4"></div>
                   <div className="h-4 bg-slate-100 rounded w-5/6"></div>
                 </div>
               ) : (
                <div className="relative z-10">
                  <div className="text-sm text-slate-700 leading-relaxed mb-4">
                    {!data?.Retention_Strategy ? (
                      "Adjust metrics to recalculate strategy."
                    ) : (
                      <ul className="space-y-3">
                        {data.Retention_Strategy
                          .split(/(?:\n|•)/)
                          .map(s => s.trim())
                          .filter(s => s.length > 0 && s !== '-' && s !== '*')
                          .map((item, idx) => (
                            <li key={idx} className="flex gap-2 items-start">
                              <span className="text-teal-500 font-bold mt-0.5">•</span>
                              <span>{item.replace(/^- /, '').replace(/^\* /, '')}</span>
                            </li>
                          ))}
                      </ul>
                    )}
                  </div>
                  
                  {/* Human-in-the-Loop Override (GAP-07: now functional) */}
                  <div className="pt-4 border-t border-slate-100">
                    {!showOverrideForm ? (
                      <div className="flex items-center justify-between">
                        <p className="text-xs text-slate-500">Is this AI assessment inaccurate?</p>
                        <button
                          id="btn-override-ai"
                          onClick={() => setShowOverrideForm(true)}
                          className="flex items-center gap-1.5 px-3 py-1 text-xs font-medium text-slate-500 hover:text-error bg-slate-50 hover:bg-red-50 rounded border border-transparent hover:border-red-100 transition-all"
                        >
                          <ThumbsDown className="w-3 h-3" />
                          Override AI
                        </button>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <textarea
                          id="override-reason"
                          value={overrideReason}
                          onChange={(e) => setOverrideReason(e.target.value)}
                          placeholder="Explain why you disagree (min 10 chars)..."
                          className="w-full h-20 bg-white border border-slate-300 rounded-lg p-2 text-sm text-slate-900 placeholder:text-slate-400 resize-none focus:outline-none focus:border-error"
                        />
                        <div className="flex gap-2 justify-end">
                          <button
                            onClick={() => { setShowOverrideForm(false); setOverrideReason(''); }}
                            className="px-3 py-1 text-xs text-slate-500 hover:text-slate-700"
                          >
                            Cancel
                          </button>
                          <button
                            id="btn-submit-override"
                            onClick={handleOverrideSubmit}
                            disabled={overrideReason.length < 10 || overrideMutation.isPending}
                            className="px-3 py-1 text-xs font-bold bg-error hover:bg-error-dark disabled:opacity-40 text-white rounded transition-colors"
                          >
                            {overrideMutation.isPending ? 'Saving...' : 'Submit Override'}
                          </button>
                        </div>
                        {overrideMutation.isSuccess && (
                          <p className="text-xs text-emerald-600">✓ Override recorded successfully.</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
             )}
          </Card>
        </div>
      </div>
    </motion.div>
  );
}

