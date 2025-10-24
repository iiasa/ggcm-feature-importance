# Folder Structure

This is an example for the folders and files after analysis. Initially, `generated/` and `output/` are empty. The folders are populated after the data processing and analysis steps respectively.

```
ggcm-feature-importance\data
├───generated
│       climate_EPIC-IIASA_maize_rf.h5
│       yield_EPIC-IIASA_maize_rf.h5
│
├───input
│   │   Beck_KG_V1_present_0p5.tif
│   │   CLIMATEID_SLP_GGCMI_LATLON.txt
│   │   CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv
│   │   co2_historical_annual_1765_2014.txt
│   │   HWSD_soil_data_on_cropland_v2.3.nc
│   │   spam2010V2r0_global_H_MAIZE_R_30mn.tif
│   │
│   ├───climate
│   │       [ISIMIP3a GGCM climate input data]
│   │
│   └───yield
│           [ISIMIP3a GGCM simulated yields]
└───output
        results.p
        stats.csv
        yield_epic.h5
```

# Input Data Sources

* Binary HPC version [^1] of ISIMIP3a climate forcing data [^2]
* ISIMIP3a simulated yields [^3]
* MAPSPAM harvested area (here, `Beck_KG_V1_present_0p5.tif`) [^4]
* Köppen-Geiger classification (here, `spam2010V2r0_global_H_MAIZE_R_30mn.tif`) [^5]
* Site, soil and other data (here, `HWSD_soil_data_on_cropland_v2.3.nc`) [^6]

[^1]: Folberth, C., Baklanov, A., Khabarov, N., Oberleitner, T., Balkovič, J. and Skalský, R., 2025. CROMES v1. 0: a flexible CROp Model Emulator Suite for climate impact assessment. Geoscientific Model Development, 18(17), pp.5759-5779.

[^2] Cucchi, M., Weedon, G.P., Amici, A., Bellouin, N., Lange, S., Schmied, H.M., Hersbach, H. and Buontempo, C., 2020. WFDE5: bias adjusted ERA5 reanalysis data for impact studies. Earth System Science Data Discussions, 2020, pp.1-32.

[^3] Jonas Jägermeyr, Tzu-Shun Lin, Sam Rabin, Juraj Balkovic, Joshua W. Elliott, Babacar Faye, Christian Folberth, Toshichika Iizumi, Atul Jain, Takahashi Kiyoshi, Wenfeng Liu, Okada Masashi, Oleksandr Mialyk, Christoph Müller, Tommaso Stella, Chenzhi Wang, Heidi Webber, Hong Yang, Florian Zabel, Katja Frieler (2024): ISIMIP3a Simulation Data from the Agriculture Sector (v1.1). ISIMIP Repository. https://doi.org/10.48364/ISIMIP.370868.1

[^4] International Food Policy Research Institute (IFPRI), 2019, "Global Spatially-Disaggregated Crop Production Statistics Data for 2010 Version 2.0", https://doi.org/10.7910/DVN/PRFF8V, Harvard Dataverse, V4 

[^5] Beck, H.E., Zimmermann, N.E., McVicar, T.R., Vergopolan, N., Berg, A. and Wood, E.F., 2018. Present and future Köppen-Geiger climate classification maps at 1-km resolution. Scientific data, 5(1), pp.1-12.

[^6] FAO, IIASA, ISSCAS ISRIC, 2012. Jrc: Harmonized world soil database (version 1.2). FAO, Rome, Italy and IIASA, Laxenburg, Austria.

[^6] Volkholz, J., Müller, C., 2020. ISIMIP3 soil input data (v1.0). ISIMIP Repository. https://doi.org/10.48364/ISIMIP.942125

## Custom Data Formats

In this version of the pipline, the files `CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv` and `co2_historical_annual_1765_2014.txt` are only used for the calcualtion of Penman–Monteith potential evapotranspiration (PET), an can be omitted if PET is not part of the analysis. However, for the sake of completeness we provide the format of these text files here:

> CLIMATE_LAT_PD_HD_PHU_ELEV_PRMT74_mai_noirr_fH_v3c.csv
```csv
CLIMATEID,YLAT,XLON,PLDOY,HRDOY,PHU,ELEV,PRMT74
288348,83.75,-36.25,152,283,200,1,1
289348,83.75,-35.75,152,283,200,1,1
...
```

For the analysis part we included slope as a feature. The file `CLIMATEID_SLP_GGCMI_LATLON.txt` contains CSV data with these columns:

```csv
YLAT,XLON,SLP
83.75,-36.25,0.0025
83.75,-35.75,0.0025
...
```
