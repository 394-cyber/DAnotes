function time_series_d = discretize_timeseries(orig_time_series)
    median_value = median(orig_time_series);
    time_series_d = orig_time_series >= median_value;

end