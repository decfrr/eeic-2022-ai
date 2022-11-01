class PrintBuffer:
    """print bufferを制御して任意のタイミングで出力するクラス

    都度出力するより全出力が出揃ってから出力してくれたほうが動画っぽくなるので.
    """

    def __init__(self):
        self.buf = ''

    def prints(self, *string, sep=' ', end='\n', flush=False):
        """print()関数のような感じでbufferに書き込む"""
        if len(string) > 0:
            for s in string[:-1]:
                self.buf += str(s) + sep
            self.buf += str(string[-1])
        self.buf += end
        if flush:
            self.flush()

    def flush(self):
        print(self.buf, end='', flush=True)
        self.clear()

    def clear(self):
        self.buf = ''
