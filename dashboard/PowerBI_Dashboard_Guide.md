# Power BI Dashboard Specification Guide

This guide provides the blueprint, DAX formulas, and design specifications required to replicate the interactive Power BI dashboard (`PowerBI_Dashboard.pbix`) for the **Healthcare Analytics Platform**.

---

## 1. Data Source & Import
* **Source Type**: CSV File
* **File Connection**: `data/healthcare_dataset.csv`
* **Data Types Checklist**:
  - `Patient_ID`: Text (Key)
  - `Age`: Whole Number
  - `Gender`: Text
  - `Hypertension`: Whole Number (or True/False)
  - `Heart_Disease`: Whole Number (or True/False)
  - `Admission_Date`: Date
  - `Length_of_Stay`: Whole Number
  - `Billing_Amount`: Decimal Number (Format: Currency `$`)
  - `Stroke`: Whole Number (or True/False)

---

## 2. Calculated Columns (Power Query / DAX)
Create the following calculated column in Power BI to group patient ages:

```dax
Age_Group = 
SWITCH(
    TRUE(),
    healthcare_dataset[Age] < 18, "Under 18",
    healthcare_dataset[Age] <= 30, "18-30",
    healthcare_dataset[Age] <= 45, "31-45",
    healthcare_dataset[Age] <= 60, "46-60",
    "Over 60"
)
```

---

## 3. Key DAX Measures
Create a dedicated table named `_Measures` and implement the following business formulas:

### KPI: Total Patient Count
```dax
Total Patients = COUNT(healthcare_dataset[Patient_ID])
```

### KPI: Stroke Patients
```dax
Stroke Cases = CALCULATE([Total Patients], healthcare_dataset[Stroke] = 1)
```

### KPI: Stroke Incidence Rate (%)
```dax
Stroke Rate = DIVIDE([Stroke Cases], [Total Patients], 0)
```

### KPI: Total Billing Amount ($)
```dax
Total Billing = SUM(healthcare_dataset[Billing_Amount])
```

### KPI: Average Length of Stay (Days)
```dax
Avg Length of Stay = AVERAGE(healthcare_dataset[Length_of_Stay])
```

### KPI: Average Patient Age
```dax
Avg Patient Age = AVERAGE(healthcare_dataset[Age])
```

---

## 4. Visual Layout & Theme
* **Theme**: Slate Dark Mode / Teal Accents
  - Background: `#121212` (Dark Grey)
  - Card Visuals: `#1E1E1E` (Off-black card background)
  - Accent Color 1 (Clinical/Normal): `#009688` (Teal)
  - Accent Color 2 (Stroke/Alert): `#D32F2F` (Crimson)
  - Text Color: `#FFFFFF` (White) and `#B3B3B3` (Light Grey for labels)

### Grid Configuration (Top to Bottom):
1. **Header Banner (Full Width)**: Title: "Healthcare Administrative & Clinical KPI Platform", with slicers for `Admission_Date` (Timeline Selector), `Admission_Type` (Dropdown), and `Insurance_Provider` (Dropdown).
2. **KPI Card Row (4 Cards)**:
   - Card 1: `Total Patients`
   - Card 2: `Avg Length of Stay` (formatted to `1` decimal point + "Days")
   - Card 3: `Stroke Rate` (formatted as Percentage `0.0%`)
   - Card 4: `Total Billing` (formatted as Currency `$M` or `$K`)
3. **Mid-level Demographics (Row 2)**:
   - **Gender Ratio (Donut Chart)**: Legend: `Gender`. Value: `Total Patients`. (Teal/Coral split).
   - **Age Groups (Clustered Column Chart)**: X-Axis: `Age_Group` (Sorted). Y-Axis: `Total Patients`.
   - **Risk Category distribution (Pie Chart)**: Legend: `Risk_Category`. Value: `Total Patients`.
4. **Bottom-level Trends (Row 3)**:
   - **Monthly Admissions (Line Chart)**: X-Axis: `Admission_Date` (Grouped by Month/Year). Y-Axis: `Total Patients`.
   - **Disease Distribution (Clustered Bar Chart)**: X-Axis: `Total Patients`. Y-Axis: `Medical_Condition`. Color-coded by condition billing density.
   - **Insurance Cost Breakdown (Matrix Table)**:
     - Rows: `Insurance_Provider`
     - Columns: `Admission_Type`
     - Values: `Average Billing Amount` and `Total Patients`
