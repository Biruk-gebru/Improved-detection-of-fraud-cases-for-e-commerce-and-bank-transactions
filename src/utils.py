import pandas as pd
import numpy as np

def ip_to_int(ip):
    """Convert an IP address to integer format."""
    try:
        parts = list(map(int, ip.split('.')))
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan

def get_country_from_ip(ip_int, country_df):
    """
    Get country from IP integer using range-based lookup.
    Assumes country_df has 'lower_bound_ip_address', 'upper_bound_ip_address', and 'country'.
    This can be slow if done row by row on large datasets.
    """
    # Using merge_asof is much faster for range lookups if sorted
    # But IpAddress_to_Country might have non-overlapping ranges or holes.
    # Usually, merge_asof can work if we sort by lower_bound.
    pass

def map_ip_to_country(df, country_df):
    """
    Efficiently map IP addresses to countries.
    """
    df['ip_int'] = df['ip_address'].apply(ip_to_int)
    
    # Sort country_df for merge_asof
    country_df = country_df.sort_values('lower_bound_ip_address')
    
    # Perform merge_asof
    merged_df = pd.merge_asof(
        df.sort_values('ip_int'),
        country_df,
        left_on='ip_int',
        right_on='lower_bound_ip_address',
        direction='backward'
    )
    
    # Filter where ip_int is within the range
    mask = (merged_df['ip_int'] >= merged_df['lower_bound_ip_address']) & \
           (merged_df['ip_int'] <= merged_df['upper_bound_ip_address'])
    
    merged_df.loc[~mask, 'country'] = 'Unknown'
    
    # Return to original order if needed or just return merge result
    return merged_df
