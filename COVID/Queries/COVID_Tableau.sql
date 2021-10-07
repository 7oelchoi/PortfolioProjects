/*
Queries used for Tableau Visualizations
*/

-- 1. 

Select SUM(cast(new_cases as float)) as total_cases, SUM(cast(new_deaths as float)) as total_deaths, SUM(cast(new_deaths as float))/SUM(cast(New_Cases as float))*100 as DeathPercentage
From PortfolioProject..Covid_deaths
--Where location like '%states%'
where continent is not null 
--Group By date
order by 1,2


-- 2. 

Select location, SUM(cast(new_deaths as float)) as TotalDeathCount
From PortfolioProject..Covid_deaths
Where continent is null 
and location not in ('World', 'European Union', 'International')
Group by location
order by TotalDeathCount desc


-- 3.

Select Location, Population, MAX(total_cases) as HighestInfectionCount,  Max((total_cases/cast(population as float)))*100 as PercentPopulationInfected
From PortfolioProject..Covid_deaths
--Where location like '%states%'
Group by Location, Population
order by PercentPopulationInfected desc


-- 4.
--Added column of date

Select Location, Population,date, MAX(total_cases) as HighestInfectionCount,  Max((total_cases/cast(population as float)))*100 as PercentPopulationInfected
From PortfolioProject..Covid_deaths
Group by Location, Population, date
order by PercentPopulationInfected desc