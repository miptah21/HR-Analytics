with source as (
    select * from {{ source('raw_hris', 'employee_attrition_records') }}
),

renamed as (
    select
        -- In postgres columns might be lowercased depending on dlt settings, 
        -- but dlt preserves case if quoted, or normalizes it. We will use standard names.
        age::int as age,
        attrition,
        business_travel,
        daily_rate::int as daily_rate,
        department,
        distance_from_home::int as distance_from_home,
        education::int as education,
        education_field,
        employee_count::int as employee_count,
        employee_number::int as employee_number,
        environment_satisfaction::int as environment_satisfaction,
        gender,
        hourly_rate::int as hourly_rate,
        job_involvement::int as job_involvement,
        job_level::int as job_level,
        job_role,
        job_satisfaction::int as job_satisfaction,
        marital_status,
        monthly_income::numeric as monthly_income,
        monthly_rate::int as monthly_rate,
        num_companies_worked::int as num_companies_worked,
        over_18,
        over_time,
        percent_salary_hike::numeric as percent_salary_hike,
        performance_rating::int as performance_rating,
        relationship_satisfaction::int as relationship_satisfaction,
        standard_hours::int as standard_hours,
        stock_option_level::int as stock_option_level,
        total_working_years::int as total_working_years,
        training_times_last_year::int as training_times_last_year,
        work_life_balance::int as work_life_balance,
        years_at_company::int as years_at_company,
        years_in_current_role::int as years_in_current_role,
        years_since_last_promotion::int as years_since_last_promotion,
        years_with_curr_manager::int as years_with_curr_manager
    from source
)

select * from renamed
