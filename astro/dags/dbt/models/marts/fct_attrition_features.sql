with source as (
    select * from {{ ref('stg_hris') }}
),

role_stats as (
    select job_role, avg(monthly_income) as avg_role_income
    from source
    group by 1
),

level_stats as (
    select job_level, percentile_cont(0.5) within group (order by percent_salary_hike) as median_level_hike
    from source
    group by 1
),

features as (
    select
        s.employee_number,
        s.age,
        case when s.attrition = 'Yes' then 1 else 0 end as attrition,
        s.distance_from_home,
        s.monthly_income,
        s.num_companies_worked,
        s.percent_salary_hike,
        s.total_working_years,
        s.years_at_company,
        s.years_in_current_role,
        s.years_since_last_promotion,
        s.years_with_curr_manager,
        s.over_time,
        case when s.over_time = 'Yes' then 1 else 0 end as over_time_yes,
        
        s.department,
        s.education,
        s.education_field,
        s.environment_satisfaction,
        s.gender,
        s.job_involvement,
        s.job_level,
        s.job_role,
        s.job_satisfaction,
        s.marital_status,
        s.performance_rating,
        s.relationship_satisfaction,
        s.stock_option_level,
        s.training_times_last_year,
        s.work_life_balance,

        -- Engineered Features
        case when r.avg_role_income > 0 then s.monthly_income / r.avg_role_income else 1 end as compa_ratio,
        
        -- Income Growth Gap
        s.percent_salary_hike - l.median_level_hike as income_growth_gap,
        
        -- Promotion Stagnation
        s.years_since_last_promotion::numeric / (s.years_at_company + 1) as promotion_stagnation,
        
        -- Burnout Risk
        (case when s.over_time = 'Yes' then 1 else 0 end) * s.distance_from_home::numeric / greatest(s.work_life_balance, 1) as burnout_risk,
        
        -- Manager Stability
        s.years_with_curr_manager::numeric / (s.years_at_company + 1) as manager_stability,
        
        -- Engagement Index
        (s.job_satisfaction + s.environment_satisfaction + s.relationship_satisfaction + s.job_involvement)::numeric / 4 as engagement_index,
        
        -- Career Velocity
        s.job_level::numeric / (s.total_working_years + 1) as career_velocity,
        
        -- Loyalty Index
        s.years_at_company::numeric / (s.total_working_years + 1) as loyalty_index,
        
        -- Travel Burden
        case 
            when s.business_travel = 'Non-Travel' then 0
            when s.business_travel = 'Travel_Rarely' then 1
            when s.business_travel = 'Travel_Frequently' then 2
            else 1
        end as travel_burden
        
    from source s
    left join role_stats r on s.job_role = r.job_role
    left join level_stats l on s.job_level = l.job_level
)

select * from features
