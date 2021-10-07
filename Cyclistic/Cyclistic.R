# DATA RETRIEVAL

# Installing packages

install.packages("tidyverse")
install.packages("ggplot2")
install.packages("packman")
install.packages("janitor")
install.packages("lubridate") 
install.packages("rmarkdown")

# Load libraries

library("tidyverse")
library("ggplot2")
library("packman")
library("janitor") 
library("lubridate") 
library("rmarkdown")

# Getting and setting working directory

setwd("~/Work/PortfolioProjects/Cyclistic/Datasets")

# Importing Files

Apr_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202004.csv")
May_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202005.csv")
Jun_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202006.csv")
Jul_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202007.csv")
Aug_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202008.csv")
Sep_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202009.csv")
Oct_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202010.csv")
Nov_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202011.csv")
Dec_2020 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202012.csv")
Jan_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202101.csv")
Feb_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202102.csv")
Mar_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202103.csv")
Apr_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202104.csv")
May_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202105.csv")
Jun_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202106.csv")
Jul_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202107.csv")
Aug_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202108.csv")
Sep_2021 <- read_csv("~/Work/PortfolioProjects/Cyclistic/Datasets/202109.csv")

# Looking at column names, looking for discrepancies

colnames(Apr_2020)
colnames(May_2020)
colnames(Jun_2020)
colnames(Jul_2020)
colnames(Aug_2020)
colnames(Sep_2020)
colnames(Oct_2020)
colnames(Nov_2020)
colnames(Dec_2020)
colnames(Jan_2021)
colnames(Feb_2021)
colnames(Mar_2021)
colnames(Apr_2021)
colnames(May_2021)
colnames(Jun_2021)
colnames(Jul_2021)
colnames(Aug_2021)
colnames(Sep_2021)

# Look for anomalies in data frames

str(Apr_2020)
str(May_2020)
str(Jun_2020)
str(Jul_2020)
str(Aug_2020)
str(Sept_2020)
str(Oct_2020)
str(Nov_2020)
str(Dec_2020)
str(Jan_2021)
str(Feb_2021)
str(Mar_2021)
str(Apr_2021)
str(Jun_2021)
str(Jul_2021)
str(Aug_2021)
str(Sep_2021)

# Convert ride_id and rideable_type to character so that merging works

Apr_2020 <-  mutate(Apr_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Jun_2020 <-  mutate(Jun_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Jul_2020 <-  mutate(Jul_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Aug_2020 <-  mutate(Aug_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Sep_2020 <-  mutate(Sep_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Oct_2020 <-  mutate(Oct_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Nov_2020 <-  mutate(Nov_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Dec_2020 <-  mutate(Dec_2020, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Jan_2021 <-  mutate(Jan_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Feb_2021 <-  mutate(Feb_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Mar_2021 <-  mutate(Mar_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Apr_2021 <-  mutate(Apr_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type))
May_2021 <-  mutate(May_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Jun_2021 <-  mutate(Jun_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Jul_2021 <-  mutate(Jul_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 
Aug_2021 <-  mutate(Aug_2021, ride_id = as.character(ride_id)
                    ,rideable_type = as.character(rideable_type)) 

# Convert start_station_id and end_station_id to double so that merging works

Apr_2020 <-  mutate(Apr_2020,  start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Jun_2020 <-  mutate(Jun_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Jul_2020 <-  mutate(Jul_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Aug_2020 <-  mutate(Aug_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Sep_2020 <-  mutate(Sep_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Oct_2020 <-  mutate(Oct_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Nov_2020 <-  mutate(Nov_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Dec_2020 <-  mutate(Dec_2020, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Jan_2021 <-  mutate(Jan_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Feb_2021 <-  mutate(Feb_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Mar_2021 <-  mutate(Mar_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Apr_2021 <-  mutate(Apr_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id))
May_2021 <-  mutate(May_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Jun_2021 <-  mutate(Jun_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Jul_2021 <-  mutate(Jul_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 
Aug_2021 <-  mutate(Aug_2021, start_station_id=as.double(start_station_id)
                    ,end_station_id=as.double(end_station_id)) 

# Stitching all dataframes together

bike_trips <- rbind(Apr_2020, Jun_2020, Jul_2020, Aug_2020, Sep_2020, Oct_2020, 
                    Nov_2020, Dec_2020, Jan_2021, Feb_2021, Mar_2021, Apr_2021,
                    May_2021, Jun_2021, Jul_2021, Aug_2021)



# DATA CLEANING

# Check for anomalies
dim(bike_trips)

colnames(bike_trips) 

str(bike_trips)

summary(bike_trips)

head(bike_trips) 


# Create sub-data frames

# Data frame without data on latitude and longitude
trips <- bike_trips %>%  
  select(-c(start_lat, start_lng, end_lat, end_lng ))

# See total numbers of casual and members
table(bike_trips$member_casual) # casual = 2962392, member = 3552302

casual_percentage <- 2962392/(2962392+3552302)*100
casual_percentage # 45.47%

member_percentage <- 3552302/(2962392+3552302)*100
member_percentage # 54.53%

# Split the started_at column into: date, year,month, day, day of the week
trips$date <- as.Date(trips$started_at) #The default format is yyyy-mm-dd
trips$month <- format(as.Date(trips$date), "%m")
trips$day <- format(as.Date(trips$date), "%d")
trips$year <- format(as.Date(trips$date), "%Y")
trips$day_of_week <- format(as.Date(trips$date), "%A")

# Create column length of ride
trips$ride_length <- difftime(trips$ended_at,trips$started_at)

min_ride <- min(trips$ride_length) # Min Ride is negative

# Remove all entries with negative ride durations
trips <- trips[trips$ride_length > 0, ]

min_ride <- min(trips$ride_length) # Min Ride now 1 sec

nrow(trips) # Still 6 million rows of data

# Get rid of rows with NA values
trips <- trips[complete.cases(trips), ]

nrow(trips) # More than 3 million rows still. It's enough, but need to be
#  careful, that bias has not been added

# See total numbers of casual and members after Cleaning for bias check
table(trips['member_casual']) # casual = 1519119 member = 1844065

clean_casual_percentage <- 1519119/(1519119+1844065)*100
clean_casual_percentage # 45.12%
clean_member_percentage <- 1844065/(1519119+1844065)*100
clean_member_percentage # 54.83%
# Deviation of >0.4% -> No bias introduced during cleaning



# DATA ANALYSIS


# Pie chart plot of casual and member users
total_users <- c(1519119, 1844065)
pie(total_users)
pie_labels <- c("casual", "member")
colors <- c("brown3", "darkgreen")
pie(total_users, label=pie_labels, main = "Total Users", col=colors)
legend("topright", pie_labels, fill = colors)

# Comparing members and casual users

# Based on ride duration
mean_length <- aggregate(trips$ride_length ~ trips$member_casual, FUN = mean)
median_length <- aggregate(trips$ride_length ~ trips$member_casual, FUN = median)
max_length <- aggregate(trips$ride_length ~ trips$member_casual, FUN = max)
min_length <-aggregate(trips$ride_length ~ trips$member_casual, FUN = min)

# Based on day of week 
trips$day_of_week <- factor(trips$day_of_week, levels= c("Sunday", "Monday", 
                                         "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"))

trips[order(trips$day_of_week), ]

trips %>% 
  group_by(member_casual, day_of_week) %>% 
  summarise(number_of_rides = n()
            ,Average_Duration = mean(ride_length)) %>% 
  arrange(member_casual, day_of_week)  %>% 
  ggplot(aes(x = day_of_week, y = Average_Duration, fill = member_casual )) +
  geom_col(position = "dodge") +scale_fill_manual("member_casual",values = c("brown3", "darkgreen")) +
  theme(panel.spacing = unit(4, "lines"), axis.text.x=element_text(angle=35, hjust=1))

# Based on number of rides
trips %>% 
  na.omit(trips)%>%
  group_by(member_casual, day_of_week) %>% 
  summarise(Num_of_Rides = n()
            ,average_duration = mean(ride_length)) %>% 
  arrange(member_casual, day_of_week)  %>% 
  ggplot(aes(x = day_of_week, y = Num_of_Rides, fill = member_casual)) +
  geom_col(position = "dodge")+
  scale_fill_manual("member_casual",values = c("brown3", "darkgreen")) +
  theme(panel.spacing = unit(4, "lines"), axis.text.x=element_text(angle=0, hjust=1))

# Based on seasons
trips_seasons <- trips %>%
  mutate(Season = case_when(
    trips$month == "10" ~ "Fall",
    trips$month == "11" ~ "Fall",
    trips$month == "12" ~ "Fall",
    trips$month == "01" ~ "Winter",
    trips$month == "02" ~ "Winter",
    trips$month == "03" ~ "Winter",
    trips$month == "04" ~ "Spring",
    trips$month == "05" ~ "Spring",
    trips$month == "06" ~ "Spring",
    trips$month == "07" ~ "Summer",
    trips$month == "08" ~ "Summer",
    trips$month == "09" ~ "Summer"))
  
# Side by side comparison
ggplot(data=trips_seasons) +
  geom_bar(mapping=aes(x=Season), fill="darkgreen") + 
  facet_wrap(~member_casual)

# On top of each other
ggplot(data=trips_seasons)+
  geom_bar(mapping=aes(x=Season,fill=member_casual)) +
  scale_fill_manual("member_casual",values = c("brown3", "darkgreen"))
  

# Duration of ride based on bike type
trips %>% 
  group_by(member_casual, rideable_type) %>% 
  summarise(Number_of_Rides = n()
            ,average_duration = mean(ride_length)) %>% 
  arrange(member_casual, rideable_type)  %>% 
  ggplot(aes(x = rideable_type, y = average_duration, fill = member_casual)) +
  geom_col(width = 0.5, position = position_dodge(width = 0.5)) + labs(title = "Average ride duration vs. Ride type") +
  scale_y_continuous(labels = function(x) format(x, scientific = FALSE))+
  scale_fill_manual("member_casual",values = c("brown3", "darkgreen"))





