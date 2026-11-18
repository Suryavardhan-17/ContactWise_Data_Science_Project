# Dataset access and provenance

Source: **UCI Machine Learning Repository — Bank Marketing (dataset 222)**
https://archive.ics.uci.edu/dataset/222/bank+marketing

Original authors / related paper:
S. Moro, P. Cortez, and P. Rita. "A Data-Driven Approach to Predict the Success
of Bank Telemarketing." Decision Support Systems (2014).

The UCI entry identifies the dataset license as **CC BY 4.0**.

Run:

```bash
python -m contactwise.download
```

This extracts `bank-additional-full.csv` (~41,188 ordered records) into
`data/raw/`. The downloaded raw file is deliberately not included in GitHub.
You may also download the archive manually and put `bank-additional-full.csv`
in `data/raw/`.

The UCI dataset was collected from roughly 2008–2010; it is not live customer
data, cannot provide contemporary campaign rates, and does not contain unique
client identifiers or an untreated control group.
