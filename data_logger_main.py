# -*- coding: utf-8 -*-
"""
data_logger_main.py
------------------
Real-time Market Data Logger for KRX-NXT Arbitrage System

Logs real-time market data from 713 symbols (both KRX and NXT venues)
to a single daily Parquet file for later simulation use.

Design:
- Captures "주식시세" real-time data only
- Uses screen sharding (8 screens, ~89 symbols each)
- Buffers 5,000 records before writing
- Outputs to single daily Parquet file
- Independent from main arbitrage system
"""

import sys
import logging
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import QObject, pyqtSignal, QTimer, QThread
from PyQt5.QAxContainer import QAxWidget

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


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

        # Initialize Kiwoom OCX
        self.ocx = QAxWidget("KHOPENAPI.KHOpenAPICtrl.1")
        self.is_connected = False

        # Screen management
        self.screen_numbers = ["1000", "1001", "1002", "1003", "1004", "1005", "1006", "1007"]
        self.screen_symbol_map = {}  # screen_no -> [symbols]
        self.symbol_screen_map = {}  # symbol -> screen_no

        # Real-time data FIDs (matching arbitrage system)
        self.quote_fids = "10;11;12;13;27;28"  # 현재가, 전일대비, 등락률, 누적거래량, 매도호가, 매수호가

        # Connect events
        self._connect_events()

        logger.info("KiwoomDataLogger initialized")

    def _connect_events(self):
        """Connect Kiwoom OCX events"""
        try:
            self.ocx.OnEventConnect.connect(self._on_event_connect)
            self.ocx.OnReceiveRealData.connect(self._on_receive_real_data)
            logger.info("Kiwoom events connected")
        except Exception as e:
            logger.error(f"Failed to connect events: {e}")
            raise

    def connect_to_server(self):
        """Connect to Kiwoom server"""
        try:
            ret = self.ocx.CommConnect()
            if ret == 0:
                self.status_update.emit("연결 요청 중...")
                logger.info("Connection request sent")
            else:
                self.status_update.emit(f"연결 실패: {ret}")
                logger.error(f"Connection failed: {ret}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.status_update.emit(f"연결 오류: {e}")

    def _on_event_connect(self, err_code):
        """Handle connection event"""
        if err_code == 0:
            self.is_connected = True
            self.connected.emit(True)
            self.status_update.emit("키움 서버 연결 성공")
            logger.info("Connected to Kiwoom server")
        else:
            self.is_connected = False
            self.connected.emit(False)
            self.status_update.emit(f"연결 실패: {err_code}")
            logger.error(f"Connection failed: {err_code}")

    def load_symbols_and_register(self, symbols: List[str]):
        """Load symbols and register for real-time data"""
        if not self.is_connected:
            logger.error("Not connected to Kiwoom")
            return False

        try:
            # Distribute symbols across screens
            symbols_per_screen = len(symbols) // len(self.screen_numbers) + 1

            for i, screen_no in enumerate(self.screen_numbers):
                start_idx = i * symbols_per_screen
                end_idx = min(start_idx + symbols_per_screen, len(symbols))
                screen_symbols = symbols[start_idx:end_idx]

                if not screen_symbols:
                    continue

                # Store mapping
                self.screen_symbol_map[screen_no] = screen_symbols
                for symbol in screen_symbols:
                    self.symbol_screen_map[symbol] = screen_no

                # Register KRX symbols
                krx_codes = ";".join(screen_symbols)
                ret_krx = self.ocx.SetRealReg(screen_no, krx_codes, self.quote_fids, "0")

                # Register NXT symbols (with _NX suffix)
                nxt_codes = ";".join([f"{symbol}_NX" for symbol in screen_symbols])
                ret_nxt = self.ocx.SetRealReg(screen_no, nxt_codes, self.quote_fids, "0")

                logger.info(f"Screen {screen_no}: {len(screen_symbols)} symbols - KRX:{ret_krx}, NXT:{ret_nxt}")

                if ret_krx != 0:
                    print(f"WARNING: KRX registration failed for screen {screen_no}: {ret_krx}")
                if ret_nxt != 0:
                    print(f"WARNING: NXT registration failed for screen {screen_no}: {ret_nxt}")

            total_registered = len(symbols) * 2  # KRX + NXT
            self.status_update.emit(f"실시간 등록 완료: {total_registered}개 심볼")
            logger.info(f"Real-time registration complete: {total_registered} symbols")
            return True

        except Exception as e:
            logger.error(f"Symbol registration error: {e}")
            print(f"ERROR: Symbol registration failed: {e}")
            return False

    def _on_receive_real_data(self, code: str, real_type: str, real_data: str):
        """Handle real-time data reception"""
        try:
            # Only process 주식시세 data (matching arbitrage system)
            if real_type != "주식시세":
                return

            # Determine venue and symbol
            if code.endswith("_NX"):
                venue = "NXT"
                symbol = code[:-3]  # Remove _NX suffix
            else:
                venue = "KRX"
                symbol = code

            # Extract FID data
            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "venue": venue,
                "real_type": real_type,
                "fid_10": self._get_real_data(code, 10),  # 현재가
                "fid_11": self._get_real_data(code, 11),  # 전일대비
                "fid_12": self._get_real_data(code, 12),  # 등락률
                "fid_13": self._get_real_data(code, 13),  # 누적거래량
                "fid_27": self._get_real_data(code, 27),  # 매도호가
                "fid_28": self._get_real_data(code, 28),  # 매수호가
                "raw_code": code
            }

            # Emit data signal
            self.data_received.emit(record)

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


class DataBuffer(QObject):
    """
    Data buffer and Parquet writer.
    Buffers incoming data and writes to Parquet files.
    """

    buffer_full = pyqtSignal(int)
    file_written = pyqtSignal(str, int)

    def __init__(self, buffer_size=5000, output_dir="./data"):
        super().__init__()

        self.buffer_size = buffer_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Data buffer
        self.buffer = []
        self.total_records = 0

        # Timer for periodic writes
        self.write_timer = QTimer()
        self.write_timer.timeout.connect(self._periodic_write)
        self.write_timer.start(30000)  # Write every 30 seconds

        logger.info(f"DataBuffer initialized: buffer_size={buffer_size}, output_dir={output_dir}")

    def add_record(self, record: Dict[str, Any]):
        """Add record to buffer"""
        self.buffer.append(record)
        self.total_records += 1

        # Write if buffer is full
        if len(self.buffer) >= self.buffer_size:
            self._write_buffer()
            self.buffer_full.emit(len(self.buffer))

    def _periodic_write(self):
        """Periodic write (every 30 seconds)"""
        if self.buffer:
            self._write_buffer()

    def _write_buffer(self):
        """Write buffer to Parquet file"""
        if not self.buffer:
            return

        try:
            # Create DataFrame
            df = pd.DataFrame(self.buffer)

            # Generate filename
            today = datetime.now().strftime("%Y%m%d")
            filename = f"market_data_{today}.parquet"
            filepath = self.output_dir / filename

            # Write or append to Parquet
            if filepath.exists():
                # Append to existing file
                existing_df = pd.read_parquet(filepath)
                combined_df = pd.concat([existing_df, df], ignore_index=True)
                combined_df.to_parquet(filepath, index=False)
            else:
                # Create new file
                df.to_parquet(filepath, index=False)

            # Clear buffer
            records_written = len(self.buffer)
            self.buffer.clear()

            self.file_written.emit(str(filepath), records_written)
            logger.info(f"Written {records_written} records to {filename}")

        except Exception as e:
            logger.error(f"Error writing buffer: {e}")
            print(f"ERROR: Failed to write data: {e}")

    def finalize(self):
        """Final write on shutdown"""
        self._write_buffer()


class SymbolLoader:
    """Symbol universe loader from Excel file"""

    @staticmethod
    def load_from_excel(filepath: str) -> List[str]:
        """Load symbols from Excel file"""
        try:
            df = pd.read_excel(filepath)

            # Assume first column contains symbols
            if 'Symbol' in df.columns:
                symbols = df['Symbol'].astype(str).tolist()
            else:
                symbols = df.iloc[:, 0].astype(str).tolist()

            # Clean symbols (remove NaN, whitespace)
            symbols = [s.strip() for s in symbols if pd.notna(s) and str(s).strip()]

            logger.info(f"Loaded {len(symbols)} symbols from {filepath}")
            return symbols

        except Exception as e:
            logger.error(f"Error loading symbols from {filepath}: {e}")
            print(f"ERROR: Could not load symbols: {e}")
            return []


class DataLoggerGUI(QMainWindow):
    """Simple GUI for data logger control"""

    def __init__(self):
        super().__init__()

        self.kiwoom = KiwoomDataLogger()
        self.buffer = DataBuffer()
        self.symbols = []

        # Connect signals
        self.kiwoom.connected.connect(self._on_connected)
        self.kiwoom.data_received.connect(self.buffer.add_record)
        self.kiwoom.status_update.connect(self._update_status)
        self.buffer.file_written.connect(self._on_file_written)

        self._init_ui()

    def _init_ui(self):
        """Initialize user interface"""
        self.setWindowTitle("Real-time Data Logger")
        self.setGeometry(100, 100, 800, 600)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Control buttons
        button_layout = QHBoxLayout()

        self.connect_btn = QPushButton("키움 연결")
        self.connect_btn.clicked.connect(self.kiwoom.connect_to_server)
        button_layout.addWidget(self.connect_btn)

        self.load_symbols_btn = QPushButton("심볼 로드 & 등록")
        self.load_symbols_btn.clicked.connect(self._load_and_register_symbols)
        self.load_symbols_btn.setEnabled(False)
        button_layout.addWidget(self.load_symbols_btn)

        self.stop_btn = QPushButton("로깅 중지")
        self.stop_btn.clicked.connect(self._stop_logging)
        self.stop_btn.setEnabled(False)
        button_layout.addWidget(self.stop_btn)

        layout.addLayout(button_layout)

        # Status display
        self.status_label = QLabel("상태: 준비")
        layout.addWidget(self.status_label)

        # Statistics
        self.stats_label = QLabel("통계: 0개 레코드")
        layout.addWidget(self.stats_label)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setMaximumHeight(300)
        layout.addWidget(self.log_display)

        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self._update_stats)
        self.update_timer.start(1000)  # Update every second

    def _on_connected(self, success: bool):
        """Handle connection result"""
        if success:
            self.connect_btn.setEnabled(False)
            self.load_symbols_btn.setEnabled(True)
            self._log_message("키움 서버 연결 성공")
        else:
            self._log_message("키움 서버 연결 실패")

    def _load_and_register_symbols(self):
        """Load symbols and register for real-time data"""
        try:
            # Load symbols from Excel (placeholder path)
            symbol_file = "./config/symbol_universe.xlsx"
            self.symbols = SymbolLoader.load_from_excel(symbol_file)

            if not self.symbols:
                self._log_message(f"ERROR: No symbols loaded from {symbol_file}")
                # Use test symbols if file not found
                self.symbols = ["005930", "000660", "035420"]  # Samsung, Hynix, Naver
                self._log_message(f"Using test symbols: {self.symbols}")

            # Register symbols
            if self.kiwoom.load_symbols_and_register(self.symbols):
                self.load_symbols_btn.setEnabled(False)
                self.stop_btn.setEnabled(True)
                self._log_message(f"실시간 데이터 등록 완료: {len(self.symbols)}개 심볼")
            else:
                self._log_message("심볼 등록 실패")

        except Exception as e:
            self._log_message(f"ERROR: {e}")

    def _stop_logging(self):
        """Stop logging and finalize"""
        self.buffer.finalize()
        self.stop_btn.setEnabled(False)
        self._log_message("로깅 중지 및 데이터 저장 완료")

    def _update_status(self, message: str):
        """Update status label"""
        self.status_label.setText(f"상태: {message}")

    def _update_stats(self):
        """Update statistics display"""
        total_records = self.buffer.total_records
        buffer_size = len(self.buffer.buffer)
        self.stats_label.setText(f"통계: {total_records}개 레코드 (버퍼: {buffer_size})")

    def _on_file_written(self, filepath: str, record_count: int):
        """Handle file write completion"""
        self._log_message(f"파일 저장: {Path(filepath).name} ({record_count}개 레코드)")

    def _log_message(self, message: str):
        """Add message to log display"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_display.append(f"[{timestamp}] {message}")
        print(f"[{timestamp}] {message}")  # Also print to console

    def closeEvent(self, event):
        """Handle window close"""
        self.buffer.finalize()
        event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)

    # Create and show GUI
    window = DataLoggerGUI()
    window.show()

    logger.info("Data Logger started")

    # Run application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()