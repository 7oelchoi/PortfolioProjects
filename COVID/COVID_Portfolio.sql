Select *
From PortfolioProject..Covid_deaths
Where continent is not null
order by 3,4

-- Select *
-- From PortfolioProject..Covid_Vaccinations$
-- order by 3,4

-- Select Data that we are going to be using
Select Location, date, total_cases, new_cases, total_deaths, population
From PortfolioProject..Covid_deaths
Where continent is not null
order by 1,2


-- Looking at Total Cases vs Total Deaths
-- Shows likelihood of dying if you are infected with covid in germany
Select Location, date, total_cases, total_deaths, (convert(float,total_deaths)/convert(float,total_cases))*100 as DeathPercentage
From PortfolioProject..Covid_deaths
Where continent is not null 
and location like '%germany%'
order by 1,2


-- Looking at Total Cases vs Population
-- Shows what percentage of population in Germany got Covid
Select Location, date, population, total_cases, (convert(float,total_cases)/convert(float,population))*100 as InfectionRate 
From PortfolioProject..Covid_deaths
where location like '%germany%'
and continent is not null 
order by 1,2


-- Looking at countries with highest infection rate compared to population
Select Location, Population, MAX(convert(float,total_cases)) as HighestInfectionCount , MAX(convert(float,total_cases)/convert(float,population))*100 as InfectionRate
From PortfolioProject..Covid_deaths
Where continent is not null 
Group by Location, Population
order by InfectionRate desc


-- Showing the countries with highest death count per population
Select Location, MAX(cast(total_deaths as float)) as TotalDeathCount
From PortfolioProject..Covid_deaths
Where continent is not null 
Group by Location
order by TotalDeathCount desc


-- Showing the continents with the highest death count per population
Select Continent, MAX(cast(total_deaths as float)) as TotalDeathCount
From PortfolioProject..Covid_deaths
Where continent is not null 
Group by Continent
order by TotalDeathCount desc


-- GLOBAL NUMBERS

-- Global numbers for total_cases, total_deaths and DeathPercentage by date
Select date, SUM(cast(new_cases as float)) as total_cases, SUM(cast(new_deaths as float)) as total_deaths,
	SUM(cast(New_deaths as float))/SUM(cast(new_cases as float))*100 as DeathPercentage
From PortfolioProject..Covid_deaths
Where continent is not null 
Group by date
order by 1,2


-- Global numbers for total_cases, total_deaths and DeathPercentage
Select SUM(cast(new_cases as float)) as total_cases, SUM(cast(new_deaths as float)) as total_deaths,
	SUM(cast(New_deaths as float))/SUM(cast(new_cases as float))*100 as DeathPercentage
From PortfolioProject..Covid_deaths
Where continent is not null 
order by 1,2


-- Looking at Total Population vs Vaccinations
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
	SUM(cast(vac.new_vaccinations as float)) 
	OVER (Partition by dea.location Order by dea.location, dea.date) as Incrementing_People_Vaccinated
	--(Incrementing_People_Vaccinated/population)*100
From PortfolioProject..Covid_deaths dea
Join PortfolioProject..Covid_Vaccinations$ vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null 
order by 2,3


-- USING CTE
With PopvsVac (Continent, Location, Date, Population, new_vaccinations, Incrementing_People_Vaccinated)
as
(
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
	SUM(cast(vac.new_vaccinations as float)) 
	OVER (Partition by dea.location Order by dea.location, dea.date) as Incrementing_People_Vaccinated
From PortfolioProject..Covid_deaths dea
Join PortfolioProject..Covid_Vaccinations$ vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null 
)
Select *, (Incrementing_People_Vaccinated/population)*100
From PopvsVac


-- TEMP TABLE

DROP Table if exists #PercentPopulationVaccinated
Create Table #PercentPopulationVaccinated
(
Continent nvarchar(255),
Location nvarchar(255),
Date datetime,
Population numeric,
New_vaccinations numeric,
Incrementing_People_Vaccinated numeric
)

Insert into #PercentPopulationVaccinated
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
	SUM(cast(vac.new_vaccinations as float)) 
	OVER (Partition by dea.location Order by dea.location, dea.date) as Incrementing_People_Vaccinated
From PortfolioProject..Covid_deaths dea
Join PortfolioProject..Covid_Vaccinations$ vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null 

Select *, (Incrementing_People_Vaccinated/population)*100
From #PercentPopulationVaccinated


-- Creating View to store data for later visualizations

Create View PercentPopulationVaccinated as
Select dea.continent, dea.location, dea.date, dea.population, vac.new_vaccinations,
	SUM(cast(vac.new_vaccinations as float)) 
	OVER (Partition by dea.location Order by dea.location, dea.date) as Incrementing_People_Vaccinated
From PortfolioProject..Covid_deaths dea
Join PortfolioProject..Covid_Vaccinations$ vac
	On dea.location = vac.location
	and dea.date = vac.date
Where dea.continent is not null 


Select *
From PercentPopulationVaccinated
