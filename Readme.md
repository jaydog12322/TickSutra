# Real-time Market Data Logger

A standalone real-time market data logger for capturing KRX and NXT stock data for arbitrage system simulation.

## Features

- **713 Symbol Support**: Captures data from all symbols listed on both KRX and NXT
- **Screen Sharding**: Distributes symbols across 8 screens (≤100 symbols per screen)
- **Dual Venue Capture**: Records both KRX and NXT data simultaneously
- **Efficient Storage**: Buffers data and writes to single daily Parquet files
- **Independent Operation**: Completely separate from main arbitrage system
- **Simple GUI**: Easy-to-use interface for control and monitoring

## Architecture

```
Excel Symbol List → Symbol Loader → Screen Assignment → 
Kiwoom SetRealReg → OnReceiveRealData → Data Buffer → 
Batched Parquet Write (daily files)
```

## Data Schema

Each record contains:
```python
{
    "timestamp": "2025-09-06T09:00:30.123456",  # Microsecond precision
    "symbol": "005930",                         # Base symbol
    "venue": "KRX" | "NXT",                    # Trading venue
    "real_type": "주식호가잔량",                 # Kiwoom real-time type
    "fid_10": 84500,    # 현재가 (Current Price)
    "fid_11": 10,       # 전일대비 (Change from Prev Close)
    "fid_12": 0.12,     # 등락율 (Rate of Change)
    "fid_13": 15000,    # 누적거래량 (Accumulated Volume)
    "fid_27": 84510,    # 매도호가1 (Best Ask Price)
    "fid_28": 84490,    # 매수호가1 (Best Bid Price)
    "fid_41": 100,      # 매도호가수량1 (Best Ask Size)
    "fid_51": 120,      # 매수호가수량1 (Best Bid Size)
    "raw_code": "005930" | "005930_NX"  # Original code from Kiwoom
}
```

## Prerequisites

### System Requirements
- Windows OS (required for Kiwoom OpenAPI+)
- **Python 3.8 (32-bit)**
- Valid Kiwoom Securities account
- Kiwoom OpenAPI+ installed

### Python Packages
Use the pinned versions below for a tested environment.  Run the helper script
so the correct `requirements.txt` is located automatically regardless of your
current working directory:
```bash
python install_requirements.py
```
This installs:
```
PyQt5==5.15.7
pandas==1.3.5
numpy==1.21.6
fastparquet==0.6.2
openpyxl==3.0.10
```

Only the packages above are required for running the logger.  Extraneous
entries from previous `pip freeze` outputs have been trimmed to keep
`requirements.txt` focused on essential dependencies.


## Setup Instructions

### 1. Install Dependencies
```bash
python install_requirements.py
```

### 2. Create Directory Structure
```
data_logger/
├── config/
│   └── symbol_universe.xlsx    # Your symbol list
├── data/                       # Output Parquet files
├── logs/                       # Log files
├── data_logger_main.py         # Main application
├── create_sample_symbols.py    # Helper to create sample symbols
└── run_logger.py              # Runner script
```

### 3. Prepare Symbol Universe
Create `./config/symbol_universe.xlsx` with your 713 symbols:

| Symbol |
|--------|
| 005930 |
| 000660 |
| 035420 |
| ...    |

Or run the helper script to create a sample file:
```bash
python create_sample_symbols.py
```

### 4. Configure Kiwoom
- Install Kiwoom OpenAPI+
- Login to Kiwoom HTS once to verify account
- Ensure account password is set in Kiwoom

## Usage

### Method 1: Simple Runner
```bash
python run_logger.py
```

### Method 2: Direct Execution
```bash
python data_logger_main.py
```

### GUI Operation
1. **Click "키움 연결"** - Connect to Kiwoom server
2. **Click "심볼 로드 & 등록"** - Load symbols and register for real-time data
3. **Monitor data flow** - Watch statistics and log messages
4. **Click "로깅 중지"** - Stop logging and finalize data files

## Output Files

Data is saved to: `./data/market_data_YYYYMMDD.parquet`

Example: `market_data_20250906.parquet`

### Reading Output Data
```python
import pandas as pd

# Load daily data
df = pd.read_parquet('./data/market_data_20250906.parquet')

# Analyze concurrent opportunities
same_second = df.groupby(df['timestamp'].str[:19]).size()
print(f"Max symbols active in same second: {same_second.max()}")

# Filter by venue
krx_data = df[df['venue'] == 'KRX']
nxt_data = df[df['venue'] == 'NXT']
```

## Configuration

### Buffer Settings
- **Buffer Size**: 5,000 records (configurable)
- **Write Frequency**: Every 30 seconds
- **Output Format**: Parquet (compressed)

### Screen Allocation
- **8 screens**: 1000-1007
- **~89 symbols per screen** (713 ÷ 8)
- **Both venues**: Each symbol registered for KRX + NXT

### Error Handling
- **Connection failures**: Logged to console, continue with available data
- **Symbol registration failures**: Warning printed, continue with successful symbols
- **Data loss tolerance**: 30-second batching (acceptable for pilot)

## Monitoring

### Real-time Statistics
- Total records captured
- Current buffer size
- Files written
- Connection status

### Log Messages
- Connection events
- Symbol registration results
- File write operations
- Error conditions

## Troubleshooting

### Common Issues

**"Connection failed"**
- Ensure Kiwoom OpenAPI+ is installed
- Check Kiwoom HTS login status
- Verify account permissions

**"Symbol registration failed"**
- Some symbols may be invalid/delisted
- Check symbol format (6-digit codes)
- Verify KRX/NXT listing status

**"No data received"**
- Market may be closed
- Check time windows (09:00-15:20 for main session)
- Verify real-time data subscription

**"File write errors"**
- Check disk space
- Verify write permissions to ./data/ directory
- Ensure no antivirus blocking file creation

### Debug Mode
Enable debug logging by modifying the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Data Analysis Examples

### Basic Statistics
```python
import pandas as pd

df = pd.read_parquet('./data/market_data_20250906.parquet')

print(f"Total records: {len(df)}")
print(f"Unique symbols: {df['symbol'].nunique()}")
print(f"Venues: {df['venue'].value_counts()}")
print(f"Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
```

### Concurrent Opportunities
```python
# Group by second to find concurrent activity
df['second'] = df['timestamp'].str[:19]  # Extract to second precision
concurrent = df.groupby('second')['symbol'].nunique()

print(f"Max concurrent symbols in same second: {concurrent.max()}")

# Find peak activity periods
peak_activity = concurrent.nlargest(10)
print("Top 10 busiest seconds:")
print(peak_activity)
```

### Spread Analysis Preparation
```python
# Prepare data for spread analysis
def prepare_spread_data(df):
    """Prepare data for spread analysis between venues"""
    
    # Pivot data by venue for each symbol
    krx_data = df[df['venue'] == 'KRX'].set_index(['timestamp', 'symbol'])
    nxt_data = df[df['venue'] == 'NXT'].set_index(['timestamp', 'symbol'])
    
    # Merge KRX and NXT data for same timestamp/symbol
    merged = krx_data.join(nxt_data, how='inner', lsuffix='_krx', rsuffix='_nxt')
    
    # Calculate potential spreads using best bid/ask
    merged['spread_bps'] = (
        (merged['fid_28_krx'] - merged['fid_27_nxt']) / merged['fid_27_nxt'] * 10000
    )
    
    return merged

spread_data = prepare_spread_data(df)
print(f"Potential arbitrage opportunities: {len(spread_data[abs(spread_data['spread_bps']) > 5])}")
```

## Performance Expectations

### Data Volume Estimates
- **713 symbols × 2 venues = 1,426 data streams**
- **~6.5 trading hours = 23,400 seconds**
- **Estimated 50-200 ticks per symbol per hour**
- **Daily file size: 10-50 MB (compressed Parquet)**

### System Resources
- **Memory**: ~50-100 MB for buffer and GUI
- **CPU**: Low usage (event-driven)
- **Disk**: ~50 MB per day
- **Network**: Minimal (real-time data only)

## Integration with Arbitrage System

This logger creates data files that can be fed into your arbitrage system simulator:

### Simulation Workflow
1. **Record Phase**: Use this logger to capture market data
2. **Analysis Phase**: Analyze patterns and opportunities
3. **Simulation Phase**: Feed recorded data into arbitrage system
4. **Validation Phase**: Compare simulated vs actual system behavior

### Data Compatibility
The output schema exactly matches your arbitrage system's MarketDataManager expectations:
- Same FID fields (10,11,12,13,27,28,41,51)
- Same venue detection logic (_NX suffix)
- Same real-time type ("주식호가잔량")
- Compatible timestamp format

## Advanced Configuration

### Custom Buffer Settings
```python
# In data_logger_main.py, modify DataBuffer initialization:
self.buffer = DataBuffer(
    buffer_size=10000,  # Larger buffer
    output_dir="./custom_data"  # Custom directory
)
```

### Symbol Filtering
```python
# Filter symbols by market cap, volume, etc.
def filter_symbols(symbols):
    # Add your filtering logic here
    high_volume_symbols = [s for s in symbols if is_high_volume(s)]
    return high_volume_symbols
```

### Custom FID Fields
```python
# To capture additional fields, modify hoga_fids:
self.hoga_fids = "10;11;12;13;27;28;41;51;20"  # Add FID 20 (체결시간)
```

## Production Deployment

### For Extended Operation
1. **Error Recovery**: Add auto-reconnection logic
2. **Health Monitoring**: Add Slack/email alerts
3. **Data Validation**: Add real-time data quality checks
4. **Backup Strategy**: Multiple output locations
5. **Log Rotation**: Prevent log file growth

### Security Considerations
- Store Kiwoom credentials securely
- Restrict file system access
- Monitor for unusual data patterns
- Regular security updates

## Support and Maintenance

### Log Files
- Application logs: Console output
- Kiwoom events: Real-time data reception
- Error tracking: Failed registrations, write errors

### Monitoring Checklist
- [ ] Kiwoom connection status
- [ ] Symbol registration success rate
- [ ] Data reception frequency
- [ ] File write operations
- [ ] Disk space availability
- [ ] Buffer utilization

## Version History

### v1.0 (Current)
- Initial release
- Basic real-time data capture
- Parquet output format
- Simple GUI interface
- Screen sharding support

### Planned Features
- [ ] Auto-reconnection
- [ ] Data quality validation
- [ ] Performance metrics
- [ ] Configuration file support
- [ ] Advanced filtering options

## License and Disclaimer

This software is for educational and research purposes. Trading involves financial risk. Users are responsible for:
- Compliance with local regulations
- Proper risk management
- Data accuracy verification
- System security

---

**Contact**: For issues or questions, check the console output and log files first. Most problems are related to Kiwoom API setup or symbol configuration.