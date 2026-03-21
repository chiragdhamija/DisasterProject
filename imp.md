Great, you’re in good shape now.

I validated your new mapped export:
- Folder: `NextDayWildFireSpr/data/ndws64_meta_full`
- Files: `8` (`train=6`, `eval=1`, `test=1`)
- Schema: all required metadata fields present
- Metadata: valid (`sample_lon`, `sample_lat`, `sample_date`, `start_day`, `split`)

Important finding:
- Only about **20.38%** samples are inside California (`2827 / 13874`).

Since your external datasets are California-only, the next step should be:

1. Create a **CA-only filtered dataset** from `ndws64_meta_full` into a new folder (keep split names).  
2. Validate CA-only schema + counts.  
3. Generate a mapping table (`sample_id, split, date, lon, lat`) for joins.  
4. Start feature engineering joins (roads/fire/SVI/ACS) on this CA-only set.  
5. Update training pipeline to use enhanced channels.

If you want, I can do step 1 now and create `NextDayWildFireSpr/data/ndws64_meta_ca/` for you.