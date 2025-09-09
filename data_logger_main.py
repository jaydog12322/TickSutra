# -*- coding: utf-8 -*-
"""
data_logger_main.py
------------------
Real-time Market Data Logger for KRX-NXT Arbitrage System
Logs real-time market data from 713 symbols (both KRX and NXT venues)
to a single daily Parquet file for later simulation use.

MODIFIED VERSION: Uses external parquet_tool.exe for 32-bit compatibility

Design:
- Captures "주식호가잔량" level-1 order book only
- Uses screen sharding (8 screens, ~89 symbols each)
- Double-buffers 100,000 records with background writer thread
- Outputs to single daily Parquet file via external tool
- Independent from main arbitrage system
"""
import sys
import json
import logging
import time
import subprocess
import tempfile
import threading
from queue import Queue
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import QObject, pyqtSignal, QTimer
from PyQt5.QAxContainer import QAxWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DataBuffer(QObject):
    """
    MODIFIED: Data buffer using external parquet_tool.exe
    Buffers incoming data and writes to Parquet files via external tool.
    Implements double buffering and an asynchronous writer thread.
    """

    buffer_full = pyqtSignal(int)
    file_written = pyqtSignal(str, int)
    parquet_error = pyqtSignal(str)

    def __init__(self, buffer_size=100000, output_dir="./data", parquet_tool_path="./parquet_tool.exe"):
        super().__init__()

        self.buffer_size = buffer_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.parquet_tool_path = Path(parquet_tool_path)

        # Double buffers
        self.buffer: List[Dict[str, Any]] = []  # active buffer
        self.pending_buffer: List[Dict[str, Any]] = []  # buffer awaiting write
        self.total_records = 0

        # Queue and writer thread
        self._write_queue: Queue = Queue()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

        # Timer for periodic writes (every 1 second)
        self.write_timer = QTimer()
        self.write_timer.timeout.connect(self._periodic_write)
        self.write_timer.start(1000)

        # Validate parquet tool exists
        self.tool_available = self._validate_parquet_tool()

        logger.info(f"DataBuffer initialized: buffer_size={buffer_size}, output_dir={output_dir}")
        logger.info(
            f"Using parquet tool: {self.parquet_tool_path} ({'Available' if self.tool_available else 'NOT FOUND'})")

    def _validate_parquet_tool(self):
        """Validate that parquet tool exists and works"""
        if not self.parquet_tool_path.exists():
            error_msg = f"Parquet tool not found: {self.parquet_tool_path}"
            logger.error(error_msg)
            self.parquet_error.emit(error_msg)
            return False

        try:
            # Test the tool
            result = subprocess.run(
                [str(self.parquet_tool_path), "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode != 0:
                error_msg = f"Parquet tool test failed: {result.stderr}"
                logger.error(error_msg)
                self.parquet_error.emit(error_msg)
                return False

            logger.info("✓ Parquet tool validation successful")
            return True

        except Exception as e:
            error_msg = f"Parquet tool validation error: {e}"
            logger.error(error_msg)
            self.parquet_error.emit(error_msg)
            return False

    def add_record(self, record: Dict[str, Any]):
        """Add record to active buffer"""
        self.buffer.append(record)
        self.total_records += 1

        if len(self.buffer) >= self.buffer_size:
            self._swap_and_enqueue()

    def _periodic_write(self):
        """Periodic flush of active buffer"""
        if self.buffer:
            self._swap_and_enqueue()

    def _swap_and_enqueue(self):
        """Swap active buffer with pending and enqueue for writing"""
        self.buffer, self.pending_buffer = self.pending_buffer, self.buffer
        full_buffer = self.pending_buffer
        self.pending_buffer = []
        self._write_queue.put(full_buffer)
        self.buffer_full.emit(len(full_buffer))

    def _writer_loop(self):
        """Background writer thread loop"""
        while True:
            chunk = self._write_queue.get()
            if chunk is None:
                self._write_queue.task_done()
                break
            self._write_chunk(chunk)
            self._write_queue.task_done()

    def _write_chunk(self, chunk: List[Dict[str, Any]]):
        """Write a chunk of records to Parquet via external tool"""
        if not chunk:
            return

        if not self.tool_available:
            logger.warning("Parquet tool not available, skipping write")
            return

        try:
            # Generate filename
            today = datetime.now().strftime("%Y%m%d")
            filename = f"market_data_{today}.parquet"
            filepath = self.output_dir / filename

            # Create temporary JSON file for the buffer data
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as temp_file:
                json.dump(chunk, temp_file, ensure_ascii=False, indent=None)
                temp_json_path = temp_file.name

            try:
                # Decide whether to write or append
                if filepath.exists():
                    # Append to existing file
                    result = self._run_parquet_tool('append', temp_json_path, str(filepath))
                else:
                    # Create new file
                    result = self._run_parquet_tool('write', temp_json_path, str(filepath))

                if result['status'] == 'success':
                    records_written = len(chunk)

                    # Emit success signal
                    self.file_written.emit(str(filepath), records_written)
                    logger.info(f"Successfully wrote {records_written} records to {filepath}")
                    if 'size_mb' in result:
                        logger.info(
                            f"File size: {result['size_mb']} MB, Total records: {result.get('total_records', 'unknown')}")
                else:
                    error_msg = f"Parquet write failed: {result.get('message', 'Unknown error')}"
                    logger.error(error_msg)
                    self.parquet_error.emit(error_msg)

            finally:
                # Clean up temporary file
                try:
                    Path(temp_json_path).unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_json_path}: {e}")

        except Exception as e:
            error_msg = f"Buffer write error: {e}"
            logger.error(error_msg)
            self.parquet_error.emit(error_msg)

    def _run_parquet_tool(self, command: str, input_file: str, output_file: str) -> Dict[str, Any]:
        """Run the parquet tool and return parsed result"""
        try:
            cmd = [
                str(self.parquet_tool_path),
                command,
                "--input", input_file,
                "--output", output_file
            ]

            logger.debug(f"Running parquet tool: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60  # 60 second timeout
            )

            # Parse JSON output
            try:
                parsed_result = json.loads(result.stdout)
            except json.JSONDecodeError:
                # If JSON parsing fails, create error result
                parsed_result = {
                    "status": "error",
                    "message": f"Invalid JSON output from parquet tool. stdout: {result.stdout}, stderr: {result.stderr}"
                }

            if result.returncode != 0:
                # Command failed
                if parsed_result.get('status') != 'error':
                    parsed_result = {
                        "status": "error",
                        "message": f"Tool exited with code {result.returncode}. stderr: {result.stderr}"
                    }

            return parsed_result

        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "Parquet tool timeout"}
        except Exception as e:
            return {"status": "error", "message": f"Subprocess error: {e}"}

    def force_write(self):
        """Force enqueue of current active buffer"""
        if self.buffer:
            logger.info(f"Force flushing {len(self.buffer)} buffered records")
            self._swap_and_enqueue()

    def stop(self):
        """Stop the buffer and writer thread, flushing remaining data"""
        self.write_timer.stop()
        self.force_write()
        # Signal writer thread to exit
        self._write_queue.put(None)
        self._writer_thread.join()
        logger.info("DataBuffer stopped")


class KiwoomDataLogger(QObject):
    """
    Kiwoom API connector for data logging only.
    Simplified version focused purely on real-time data capture.
    """
    # Signals
    connected = pyqtSignal(bool)
    data_received = pyqtSignal(dict)
    status_update = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        # Kiwoom OCX
        self.ocx = None
        self.is_connected = False

        # Real-time registration data
        self.screens = list(range(1000, 1008))  # 8 screens: 1000-1007
        # FIDs: price, change, volume + level-1 order book (주식호가잔량)
        self.hoga_fids = "10;11;12;13;27;28;41;51"

        # Data tracking
        self.registered_symbols = []
        self.data_count = 0

        # Initialize Kiwoom
        self._init_kiwoom()

    def _init_kiwoom(self):
        """Initialize Kiwoom OCX control"""
        try:
            self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
            self.ocx.OnEventConnect.connect(self._on_event_connect)
            self.ocx.OnReceiveRealData.connect(self._on_receive_real_data)
            logger.info("Kiwoom OCX initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Kiwoom OCX: {e}")
            raise

    def connect_api(self):
        """Connect to Kiwoom server"""
        try:
            if not self.ocx:
                raise Exception("OCX not initialized")

            ret = self.ocx.CommConnect()
            if ret != 0:
                raise Exception(f"CommConnect failed: {ret}")

            logger.info("Connection request sent to Kiwoom server")

        except Exception as e:
            logger.error(f"Failed to connect to Kiwoom: {e}")
            self.connected.emit(False)
            raise

    def _on_event_connect(self, err_code):
        """Handle connection event"""
        if err_code == 0:
            self.is_connected = True
            logger.info("✓ Connected to Kiwoom server")
            self.connected.emit(True)
            self.status_update.emit("Connected")
        else:
            self.is_connected = False
            logger.error(f"✗ Connection failed: {err_code}")
            self.connected.emit(False)
            self.status_update.emit(f"Connection failed: {err_code}")

    def register_symbols(self, symbols: List[str]) -> int:
        """Register symbols for real-time data.

        Returns
        -------
        int
            Number of symbols successfully registered.
        """
        if not self.is_connected:
            logger.error("Cannot register symbols: not connected to Kiwoom")
            return 0

        self.registered_symbols = symbols
        symbols_per_screen = len(symbols) // len(self.screens) + 1
        success_count = 0

        for i, screen in enumerate(self.screens):
            start_idx = i * symbols_per_screen
            end_idx = min((i + 1) * symbols_per_screen, len(symbols))
            screen_symbols = symbols[start_idx:end_idx]

            if not screen_symbols:
                continue

            # Register both KRX and NXT versions
            all_codes = []
            for symbol in screen_symbols:
                all_codes.append(symbol)  # KRX version
                all_codes.append(f"{symbol}_NX")  # NXT version

            codes_str = ";".join(all_codes)

            try:
                ret = self.ocx.SetRealReg(
                    str(screen),  # Screen number
                    codes_str,  # Symbol codes
                    self.hoga_fids,  # FID list
                    "1"  # Real-time type
                )

                if ret == 0:
                    success_count += len(screen_symbols)
                    logger.info(
                        f"✓ Screen {screen}: Registered {len(screen_symbols)} symbols ({len(all_codes)} codes)"
                    )
                else:
                    logger.error(f"✗ Screen {screen}: Registration failed: {ret}")
                    self.status_update.emit(f"Screen {screen} registration failed: {ret}")

            except Exception as e:
                logger.error(f"Error registering screen {screen}: {e}")
                self.status_update.emit(f"Error registering screen {screen}: {e}")

        logger.info(
            f"Registration complete: {success_count}/{len(symbols)} symbols across {len(self.screens)} screens"
        )
        return success_count

    def _on_receive_real_data(self, code, real_type, data):
        """Handle real-time data"""
        try:
            # Only process level-1 order book events
            if real_type != "주식호가잔량":
                return

            # Determine venue and symbol
            if code.endswith("_NX"):
                venue = "NXT"
                symbol = code[:-3]  # Remove _NX suffix
            else:
                venue = "KRX"
                symbol = code

            # Extract FID data (price, volume, L1 order book)
            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "venue": venue,
                "real_type": real_type,
                "fid_10": self._get_real_data(code, 10),  # 현재가 (Current Price)
                "fid_11": self._get_real_data(code, 11),  # 전일대비 (Change from Prev Close)
                "fid_12": self._get_real_data(code, 12),  # 등락율 (Rate of Change)
                "fid_13": self._get_real_data(code, 13),  # 누적거래량 (Accumulated Volume)
                "fid_27": self._get_real_data(code, 27),  # 매도호가1 (Best Ask Price)
                "fid_28": self._get_real_data(code, 28),  # 매수호가1 (Best Bid Price)
                "fid_41": self._get_real_data(code, 41),  # 매도호가수량1 (Best Ask Size)
                "fid_51": self._get_real_data(code, 51),  # 매수호가수량1 (Best Bid Size)
                "raw_code": code
            }

            # Emit data signal
            self.data_received.emit(record)
            self.data_count += 1

            if self.data_count % 100 == 0:
                logger.info(f"Received {self.data_count} data points")

        except Exception as e:
            logger.error(f"Error processing real data for {code}: {e}")

    def _get_real_data(self, code: str, fid: int):
        """Get real-time data for specific FID"""
        try:
            data = self.ocx.GetCommRealData(code, fid)
            return data.strip() if data else ""
        except Exception as e:
            logger.debug(f"Error getting FID {fid} for {code}: {e}")
            return ""

    def disconnect_api(self):
        """Disconnect from Kiwoom server"""
        try:
            if self.ocx and self.is_connected:
                self.ocx.CommTerminate()
                self.is_connected = False
                logger.info("Disconnected from Kiwoom server")
                self.connected.emit(False)
                self.status_update.emit("Disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")


class SymbolLoader:
    """Symbol loader for reading Excel/CSV files"""

    @staticmethod
    def load_symbols(filepath: str) -> List[str]:
        """Load symbols from Excel file"""
        try:
            # Try Excel first
            if filepath.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(filepath)
            else:
                df = pd.read_csv(filepath)

            # Try common column names
            symbol_col = None
            for col in ['Symbol', 'symbol', 'SYMBOL', '종목코드', '코드']:
                if col in df.columns:
                    symbol_col = col
                    break

            if symbol_col is None:
                # Use first column
                symbol_col = df.columns[0]
                logger.warning(f"No standard symbol column found, using: {symbol_col}")

            # Extract only numeric portions, drop non-numeric entries,
            # and left-pad codes to six digits
            symbols_series = df[symbol_col].astype(str).str.replace(r'\D', '', regex=True)

            # Remove empty strings and codes longer than 6 digits
            symbols_series = symbols_series[(symbols_series != '') & (symbols_series.str.len() <= 6)]

            # Pad with leading zeros and filter for exact 6-digit codes
            symbols_series = symbols_series.str.zfill(6)
            symbols_series = symbols_series[symbols_series.str.len() == 6]

            symbols = symbols_series.tolist()

            logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
            return symbols

        except Exception as e:
            logger.error(f"Failed to load symbols from {filepath}: {e}")
            return []


class DataLoggerGUI(QMainWindow):
    """Main GUI for the data logger"""

    def __init__(self):
        super().__init__()
        self.init_ui()
        self.init_components()
        self.connect_signals()
        # Automatically kick off the logging pipeline once the GUI is ready
        QTimer.singleShot(0, self.start_logging)

    def init_ui(self):
        """Initialize UI components"""
        self.setWindowTitle("TickSutra Data Logger (External Parquet Tool)")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)

        # Title
        title = QLabel("TickSutra Real-time Data Logger")
        layout.addWidget(title)

        # Control buttons
        button_layout = QHBoxLayout()

        self.connect_btn = QPushButton("키움 연결")
        self.load_symbols_btn = QPushButton("심볼 로드 & 등록")
        self.start_btn = QPushButton("로깅 시작")
        self.stop_btn = QPushButton("로깅 중지")

        self.load_symbols_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)

        button_layout.addWidget(self.connect_btn)
        button_layout.addWidget(self.load_symbols_btn)
        button_layout.addWidget(self.start_btn)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Status section
        status_layout = QVBoxLayout()

        self.connection_status = QLabel("상태: 연결 안됨")
        self.symbols_status = QLabel("심볼: 0개 로드됨")
        self.parquet_status = QLabel("Parquet 도구: 확인 중...")
        self.buffer_status = QLabel("버퍼: 0/5000")
        self.records_status = QLabel("총 레코드: 0")

        status_layout.addWidget(self.connection_status)
        status_layout.addWidget(self.symbols_status)
        status_layout.addWidget(self.parquet_status)
        status_layout.addWidget(self.buffer_status)
        status_layout.addWidget(self.records_status)

        layout.addLayout(status_layout)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(300)
        self.log_display.setReadOnly(True)
        layout.addWidget(QLabel("로그:"))
        layout.addWidget(self.log_display)

    def init_components(self):
        """Initialize data components"""
        # MODIFIED: Data buffer with external parquet tool
        self.data_buffer = DataBuffer(
            buffer_size=5000,
            output_dir="./data",
            parquet_tool_path="./parquet_tool.exe"  # External tool path
        )

        # Symbol loader
        self.symbol_loader = SymbolLoader()

        # Kiwoom connection
        self.kiwoom = KiwoomDataLogger()

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_status)
        self.update_timer.start(1000)  # Update every second

        # Check parquet tool status
        QTimer.singleShot(1000, self.check_parquet_tool)

    def check_parquet_tool(self):
        """Check parquet tool status"""
        if self.data_buffer.tool_available:
            self.parquet_status.setText("Parquet 도구: ✅ 사용 가능")
        else:
            self.parquet_status.setText("Parquet 도구: ❌ 사용 불가")

    def connect_signals(self):
        """Connect all signals"""
        # Button signals
        self.connect_btn.clicked.connect(self.connect_kiwoom)
        self.load_symbols_btn.clicked.connect(self.load_symbols)
        self.start_btn.clicked.connect(self.start_logging)
        self.stop_btn.clicked.connect(self.stop_logging)

        # Kiwoom signals
        self.kiwoom.connected.connect(self.on_kiwoom_connected)
        self.kiwoom.data_received.connect(self.on_data_received)
        self.kiwoom.status_update.connect(self.on_status_update)

        # Data buffer signals
        self.data_buffer.file_written.connect(self.on_file_written)
        self.data_buffer.parquet_error.connect(self.on_parquet_error)

    def connect_kiwoom(self):
        """Connect to Kiwoom API"""
        try:
            self.kiwoom.connect_api()
            self.log_message("키움 API 연결 요청...")
        except Exception as e:
            self.log_message(f"연결 실패: {e}")

    def load_symbols(self):
        """Load symbols and register for real-time data"""
        try:
            # Load symbols from file
            symbol_file = "./config/symbol_universe.xlsx"
            symbols = self.symbol_loader.load_symbols(symbol_file)

            if not symbols:
                # Try alternative file paths
                for alt_path in ["./symbol_universe.xlsx", "./symbols.xlsx", "./config/symbols.xlsx"]:
                    if Path(alt_path).exists():
                        symbols = self.symbol_loader.load_symbols(alt_path)
                        if symbols:
                            break

            if not symbols:
                self.log_message("❌ 심볼 파일을 찾을 수 없습니다. ./config/symbol_universe.xlsx 확인")
                return

            # Register with Kiwoom and get count of successfully registered symbols
            registered = self.kiwoom.register_symbols(symbols)

            if registered > 0:
                self.symbols_status.setText(f"심볼: {registered}개 등록됨")
                self.log_message(f"✅ {registered}개 심볼 등록 완료")
                # Enable start button only when at least one symbol registered
                self.start_btn.setEnabled(True)
            else:
                self.symbols_status.setText("심볼: 0개 등록됨")
                self.log_message("❌ 심볼 등록 실패")
                self.start_btn.setEnabled(False)

        except Exception as e:
            self.log_message(f"❌ 심볼 로드 실패: {e}")

    def start_logging(self):
        """Start data logging.

        This orchestrates the full startup sequence:
        - Connect to Kiwoom if not already connected
        - Load and register symbols once connected
        - Enable logging controls
        """

        # Ensure we are connected
        if not self.kiwoom.is_connected:
            # Connection is asynchronous; on_kiwoom_connected will recall this
            self.log_message("키움 API 연결 요청...")
            self.connect_kiwoom()
            return

        # Ensure symbols are registered
        if not self.kiwoom.registered_symbols:
            self.log_message("심볼 로드 및 등록 중...")
            self.load_symbols()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.connect_btn.setEnabled(False)
        self.load_symbols_btn.setEnabled(False)
        self.log_message("✅ 로깅 시작")

    def stop_logging(self):
        """Stop data logging"""
        self.data_buffer.force_write()
        self.kiwoom.disconnect_api()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.log_message("⏹️ 로깅 중지 및 버퍼 플러시")

    def on_kiwoom_connected(self, success: bool):
        """Handle Kiwoom connection result"""
        if success:
            self.connection_status.setText("상태: ✅ 연결됨")
            # Proceed with loading symbols and starting the logger
            self.start_logging()
        else:
            self.connection_status.setText("상태: ❌ 연결 실패")

    def on_data_received(self, record: Dict[str, Any]):
        """Handle received market data"""
        self.data_buffer.add_record(record)

    def on_status_update(self, message: str):
        """Handle status updates"""
        self.log_message(f"상태: {message}")

    def on_file_written(self, filepath: str, records: int):
        """Handle file write completion"""
        filename = Path(filepath).name
        self.log_message(f"💾 파일 저장: {filename} ({records:,} 레코드)")

    def on_parquet_error(self, error_msg: str):
        """Handle parquet tool errors"""
        self.log_message(f"❌ Parquet 오류: {error_msg}")

    def update_status(self):
        """Update status displays"""
        self.buffer_status.setText(f"버퍼: {len(self.data_buffer.buffer)}/{self.data_buffer.buffer_size}")
        self.records_status.setText(f"총 레코드: {self.data_buffer.total_records:,}")

    def log_message(self, message: str):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")

        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def closeEvent(self, event):
        """Handle application close"""
        self.log_message("애플리케이션 종료 중...")
        self.data_buffer.stop()
        self.kiwoom.disconnect_api()
        event.accept()


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Create main window
    window = DataLoggerGUI()
    window.show()

    logger.info("TickSutra Data Logger started (External Parquet Tool)")

    # Start application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
