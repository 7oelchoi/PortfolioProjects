-- Data cleaning on the Nashville housing Data sheet --

SELECT *
FROM PortfolioProject.dbo.Housing

-- First we are going to standardize the Date Format
-- Currently With time, even though time is always 00:00

SELECT SaleDateConverted, CONVERT(Date, SaleDate)
FROM PortfolioProject.dbo.Housing

UPDATE Housing
	SET SaleDate = CONVERT(Date, SaleDate)

-- Update somehow does not change the column

ALTER TABLE Housing
	ADD SaleDateConverted Date;

UPDATE Housing
	SET SaleDateConverted = CONVERT(Date, SaleDate)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Populate Property Address data -- 

SELECT *
FROM PortfolioProject.dbo.Housing
WHERE PropertyAddress is NULL

-- We have NULL Values, 

SELECT *
FROM PortfolioProject.dbo.Housing
ORDER BY ParcelID

SELECT a.ParcelID, a.PropertyAddress, b.ParcelID, b.PropertyAddress, ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM PortfolioProject.dbo.Housing a
JOIN PortfolioProject.dbo.Housing b
	ON a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress is null


UPDATE a 
SET PropertyAddress = ISNULL(a.PropertyAddress, b.PropertyAddress)
FROM PortfolioProject.dbo.Housing a
JOIN PortfolioProject.dbo.Housing b
	ON a.ParcelID = b.ParcelID
	AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress is NULL

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Splitting the PropertyAddress into Individual Columns --

SELECT PropertyAddress
FROM PortfolioProject.dbo.Housing

ALTER TABLE Housing
	ADD PropertyStreet Nvarchar(255)

ALTER TABLE Housing
	ADD PropertyCity Nvarchar(255)
-- Minus one to get rid of comma
UPDATE Housing
	SET PropertyStreet = SUBSTRING(PropertyAddress, 1, CHARINDEX(',', PropertyAddress)-1)
-- Plus two to get rid of comma and space
UPDATE Housing
	SET PropertyCity = SUBSTRING(PropertyAddress, CHARINDEX(',', PropertyAddress)+2, LEN(PropertyAddress))


--Splitting the OwnerAddress into Individual Columns using PARSENAME

SELECT OwnerAddress
FROM PortfolioProject.dbo.Housing

ALTER TABLE Housing
	Add OwnerStreet Nvarchar(255)

ALTER TABLE Housing
	Add OwnerCity Nvarchar(255)

ALTER TABLE Housing
	Add OwnerState Nvarchar(255)

UPDATE Housing
	SET OwnerStreet = PARSENAME(REPLACE(OwnerAddress,',','.'),3)

UPDATE Housing
	SET OwnerCity = PARSENAME(REPLACE(OwnerAddress,',','.'),2)

UPDATE Housing
	SET OwnerState = PARSENAME(REPLACE(OwnerAddress,',','.'),1)


--------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Unify Sold as Vacant Column --

SELECT DISTINCT(SoldAsVacant), COUNT(SoldAsVacant)
FROM PortfolioProject.dbo.Housing
GROUP BY SoldAsVacant
ORDER BY 2

SELECT SoldAsVacant,	
	CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
		 WHEN SoldAsVacant = 'N' THEN 'No'
		 ELSE SoldAsVacant
		 END
FROM PortfolioProject.dbo.Housing

UPDATE Housing
	SET SoldAsVacant = CASE WHEN SoldAsVacant = 'Y' THEN 'Yes'
							WHEN SoldAsVacant = 'N' THEN 'No'
							ELSE SoldAsVacant
							END

--------------------------------------------------------------------------------------------------------------------------------------------------------------------

-- Removing Duplicate Rows --

WITH RowNumCTE AS (
SELECT *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyStreet,
				 SalePrice,
				 SaleDateConverted,
				 LegalReference
				 ORDER BY 
					ParcelID
					) Row_Num
FROM PortfolioProject.dbo.Housing
)

DELETE
FROM RowNumCTE
WHERE Row_Num > 1

WITH RowNumCTE AS (
SELECT *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyStreet,
				 SalePrice,
				 SaleDateConverted,
				 LegalReference
				 ORDER BY 
					ParcelID
					) Row_Num
FROM PortfolioProject.dbo.Housing
)

SELECT *
FROM RowNumCTE
WHERE Row_Num > 1

--------------------------------------------------------------------------------------------------------------------------------------------------------------------
-- Delete unused Columns --

 