import sys
import logging
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from data_logger_main import DataBuffer, KiwoomDataLogger

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('pipeline_tester')


def main():
    app = QApplication(sys.argv)

    buffer = DataBuffer(buffer_size=10)
    kiwoom = KiwoomDataLogger()

    def on_connected(ok: bool):
        logger.info('Connected: %s', ok)
        if ok:
            symbols = ['005930']  # Samsung Electronics as test symbol
            kiwoom.register_symbols(symbols)
            logger.info('Requested symbol registration for %s', symbols)
        else:
            logger.error('Connection failed, cannot register symbols')

    def on_data(record):
        buffer.add_record(record)
        logger.info('Data received. Buffer count=%d', len(buffer.buffer))

    def status_tick():
        logger.info('Heartbeat: buffer=%d total=%d', len(buffer.buffer), buffer.total_records)

    kiwoom.connected.connect(on_connected)
    kiwoom.data_received.connect(on_data)

    status_timer = QTimer()
    status_timer.timeout.connect(status_tick)
    status_timer.start(5000)

    try:
        kiwoom.connect_api()
    except Exception as exc:
        logger.error('Exception during connect_api: %s', exc)

    QTimer.singleShot(60000, app.quit)  # run for 60 seconds then exit
    app.exec_()


if __name__ == '__main__':
    main()