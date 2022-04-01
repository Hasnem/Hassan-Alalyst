-- Drop American Samoa since it has no data
DELETE FROM epidemiology WHERE State = 'American Samoa'
DELETE FROM hospitalizations WHERE State = 'American Samoa'
DELETE FROM vaccinations WHERE State = 'American Samoa'

-- Total Number of cases,deaths and percentage of death per cases in each state.

SELECT 
    State,
    MAX(cumulative_confirmed) AS Total_cases,
    MAX(cumulative_deceased) AS Total_deaths,
    100 * (MAX(cumulative_deceased) / MAX(cumulative_confirmed)) AS Percentage_of_Death_per_cases
FROM
    epidemiology
GROUP BY State
ORDER BY Percentage_of_Death_per_cases DESC;

-- confirmed new cases and new deaths per day and 7 days rolling average in each state.

Select State,date,new_confirmed,
avg (new_confirmed) 
over (partition by state order by state,date rows between 6 preceding and current row) as new_cases_rolling_avg,
new_deceased, avg (new_deceased) 
over (partition by state order by state,date rows between 6 preceding and current row) as new_deceased_rolling_avg
from epidemiology

-- percentage of the population infected with Covid in each state

Select State, 
round((max(cumulative_confirmed) / population)*100,2) as infection_rate
from epidemiology
group by state
order by infection_rate desc;

-- Percentage of the population fully vaccinated; received at least one dose.

SELECT 
    state,
    (MAX(cumulative_persons_fully_vaccinated) / population) * 100 AS pct_of_pop_fully_vaccinated,
    (MAX(cumulative_persons_vaccinated) / population) * 100 AS pct_of_pop_min_one_dose
FROM
    vaccinations
GROUP BY State
ORDER BY pct_of_pop_fully_vaccinated DESC;

-- New monthly cases per state, delta new monthly cases between current and previous month per state

select e.state,extract(month from e.date) AS month,
extract(year from e.date) as year,
sum(new_confirmed) as monthly_new_cases,
lag (sum(new_confirmed),1) 
over (partition by e.State order by extract(year from e.date),
extract(month from e.date)) as Previous_Month_cases,
sum(`new_confirmed`) - lag (sum(`new_confirmed`),1) 
over (partition by e.State order by extract(year from e.date),
extract(month from e.date)) as Delta_new_cases
from epidemiology e
join vaccinations v 
on e.state = v.state and e.date = v.date
group by state,extract(month from e.date),extract(year from e.date)
order by state,extract(year from e.date),extract(month from e.date)

-- State percentage of deaths, hospitalizations, fully vaccinated in July 2021, January 2022

Select e.state,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_deceased end) /e.population) *100 as July_2021_pct_deaths,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_deceased end) /e.population) *100 as Jan_2022_pct_deaths,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_hospitalized_patients end)/e.population) *100 as July_2021_pct_Patients_hospitalized,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_hospitalized_patients end)/e.population) *100 as Jan_2022_pct_Patients_hospitalized,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_persons_fully_vaccinated end)/e.population) *100 as July_2021_pct_people_fully_vaccinated,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_persons_fully_vaccinated end)/e.population) *100 as Jan_2022_pct_people_fully_vaccinated
from epidemiology e
join hospitalizations h on h.state=e.state and h.date=e.date
join vaccinations v on v.state=e.state and v.date=e.date
group by state
order by e.state


-- State percentage delta of deaths, hospitalizations, fully vaccinated in July 2021, January 2022

with Delta as (Select e.state,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_deceased end) /e.population) *100 as July_2021_pct_deaths,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_deceased end) /e.population) *100 as Jan_2022_pct_deaths,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_hospitalized_patients end)/e.population) *100 as July_2021_pct_Patients_hospitalized,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_hospitalized_patients end)/e.population) *100 as Jan_2022_pct_Patients_hospitalized,
(max(case when month(e.date)=7 and year(e.date)=2021 then cumulative_persons_fully_vaccinated end)/e.population) *100 as July_2021_pct_people_fully_vaccinated,
(max(case when month(e.date)=1 and year(e.date)=2022 then cumulative_persons_fully_vaccinated end)/e.population) *100 as Jan_2022_pct_people_fully_vaccinated
from epidemiology e
join hospitalizations h on h.state=e.state and h.date=e.date
join vaccinations v on v.state=e.state and v.date=e.date
group by state
order by e.state)
select state, (Jan_2022_pct_deaths - July_2021_pct_deaths) as delta_July_Jan_pct_deaths,
(Jan_2022_pct_Patients_hospitalized - July_2021_pct_Patients_hospitalized) as delta_July_Jan_pct_hospitalized,
(Jan_2022_pct_people_fully_vaccinated - July_2021_pct_people_fully_vaccinated) as delta_July_Jan_prce_fully_vaccinated
from Delta

-- Total number of new cases on peak day per state

WITH max_new AS
(SELECT  Date, State,
	MAX(new_confirmed) OVER (PARTITION BY State) AS number_New_Cases,
		 DENSE_RANK() OVER (PARTITION BY State ORDER BY new_confirmed desc) AS Highest_new_cases
FROM epidemiology)
SELECT State, Date, number_New_Cases
FROM max_new
WHERE Highest_new_cases = 1 
ORDER BY number_New_Cases DESC;

-- Total number of new deaths on peak day per state

SELECT  State,date, Max_new_deceased	
FROM
(SELECT  Date,State,
		 MAX(new_deceased) OVER (PARTITION BY State) AS Max_new_deceased,
         DENSE_RANK() OVER (PARTITION BY State ORDER BY new_deceased desc) AS Highest_deceaseds
		 FROM epidemiology) max_death
WHERE Highest_deceaseds = 1 
ORDER BY Max_new_deceased DESC;

-- Average spike in new case per month in New York.

WITH monthlySpike AS (SELECT  State, Date,Month(date) as month,year(date) as year, 
New_confirmed, avg(new_confirmed) OVER (PARTITION BY Month(date)) as Spike,
		 DENSE_RANK() OVER (PARTITION BY Month(date) ORDER BY New_Confirmed DESC) AS rnk
FROM epidemiology
WHERE State = 'New York')
SELECT  State,Year(date),Month(date),
		ROUND(Spike,2) as AvgSpikeInCases
FROM monthlySpike
WHERE rnk = 1
ORDER BY Date;

/* Difference between average of new cases, average of new hospitalizations in U.S. between
week 4(01/23/2022-01/29/2022) and 5(01/30/2022-02/05/2022) */

select week(e.date) week,avg(new_confirmed) as avg_new_cases, 
avg(new_confirmed)-lag(avg(new_confirmed),1) over() as Change_in_7Days_Average_cases,
avg(new_hospitalized_patients) as avg_new_hospitalized,
avg(new_hospitalized_patients)-lag(avg(new_hospitalized_patients),1) 
over() as Change_in_7Days_Average_hospitalized
from epidemiology e 
join hospitalizations h on h.state=e.state and h.date=e.date
group by week(e.date)
having week in (4,5)





