import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming data is already loaded into mouse_metadata and study_results
# Merge the data into a single DataFrame
merged_df = pd.merge(study_results, mouse_metadata, how="outer", on="Mouse ID")

# Display the number of unique mice IDs
unique_mice = merged_df['Mouse ID'].nunique()
print(f"Number of unique mice IDs: {unique_mice}")

# Check for duplicate time points for any mouse
duplicates = merged_df[merged_df.duplicated(subset=['Mouse ID', 'Timepoint'], keep=False)]
if not duplicates.empty:
    print(f"Duplicates found for the following mouse IDs:\n{duplicates}")
else:
    print("No duplicate timepoints found.")

# Remove rows with duplicate timepoints for a mouse
cleaned_df = merged_df.drop_duplicates(subset=['Mouse ID', 'Timepoint'])

# Display updated number of unique mice IDs after cleaning
updated_unique_mice = cleaned_df['Mouse ID'].nunique()
print(f"Updated number of unique mice IDs: {updated_unique_mice}")


# Group the data by the drug regimen and calculate the summary statistics for tumor volume
summary_stats = cleaned_df.groupby('Drug Regimen')['Tumor Volume (mm3)'].agg(
    mean='mean',
    median='median',
    variance='var',
    std_dev='std',
    sem='sem'
).reset_index()

# Display the summary statistics
print(summary_stats)


# Bar chart using Pandas
bar_pandas = cleaned_df.groupby('Drug Regimen').size().plot(kind='bar', title="Number of Rows per Drug Regimen")
plt.ylabel('Number of Rows')
plt.show()

# Bar chart using Matplotlib
drug_counts = cleaned_df['Drug Regimen'].value_counts()
plt.bar(drug_counts.index, drug_counts.values)
plt.title("Number of Rows per Drug Regimen")
plt.ylabel('Number of Rows')
plt.xticks(rotation=45)
plt.show()


# Pie chart using Pandas
gender_pandas = cleaned_df['Sex'].value_counts()
gender_pandas.plot(kind='pie', autopct='%1.1f%%', title="Gender Distribution", legend=False)
plt.ylabel('')
plt.show()

# Pie chart using Matplotlib
gender_counts = cleaned_df['Sex'].value_counts()
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%')
plt.title("Gender Distribution")
plt.show()


# Filter for the four treatment regimens
treatment_list = ['Capomulin', 'Ramicane', 'Infubinol', 'Ceftamin']
final_tumors = cleaned_df[cleaned_df['Drug Regimen'].isin(treatment_list)]

# Get the final timepoint for each mouse
final_tumors_last = final_tumors.groupby('Mouse ID').last().reset_index()

# Extract the final tumor volume
tumor_volumes = final_tumors_last[['Drug Regimen', 'Tumor Volume (mm3)']]

# Calculate quartiles and IQR for each treatment regimen
tumor_volumes_grouped = tumor_volumes.groupby('Drug Regimen')
quartiles = tumor_volumes_grouped['Tumor Volume (mm3)'].quantile([0.25, 0.5, 0.75]).unstack()
IQR = quartiles[0.75] - quartiles[0.25]

# Find outliers
outliers = tumor_volumes_grouped.apply(lambda x: x[(x['Tumor Volume (mm3)'] < (quartiles[0.25] - 1.5 * IQR)) | 
                                                  (x['Tumor Volume (mm3)'] > (quartiles[0.75] + 1.5 * IQR))])
print("Outliers:\n", outliers)

# Create a box plot for final tumor volume
plt.figure(figsize=(8,6))
sns.boxplot(x='Drug Regimen', y='Tumor Volume (mm3)', data=tumor_volumes, 
            whis=[5, 95], palette="Set2")
plt.title('Final Tumor Volume by Drug Regimen')
plt.show()


capomulin_data = cleaned_df[cleaned_df['Drug Regimen'] == 'Capomulin']
mouse_id = capomulin_data['Mouse ID'].unique()[0]  # Select a single mouse

# Extract data for the selected mouse
mouse_data = capomulin_data[capomulin_data['Mouse ID'] == mouse_id]

# Create line plot of tumor volume over time
plt.plot(mouse_data['Timepoint'], mouse_data['Tumor Volume (mm3)'], marker='o')
plt.title(f"Tumor Volume Over Time for Mouse {mouse_id} (Capomulin)")
plt.xlabel('Timepoint (days)')
plt.ylabel('Tumor Volume (mm3)')
plt.show()



capomulin_avg_tumor = capomulin_data.groupby('Mouse ID').agg({'Weight (g)': 'mean', 'Tumor Volume (mm3)': 'mean'}).reset_index()

# Scatter plot of mouse weight vs average tumor volume
plt.scatter(capomulin_avg_tumor['Weight (g)'], capomulin_avg_tumor['Tumor Volume (mm3)'])
plt.title('Mouse Weight vs Tumor Volume for Capomulin')
plt.xlabel('Weight (g)')
plt.ylabel('Average Tumor Volume (mm3)')
plt.show()


