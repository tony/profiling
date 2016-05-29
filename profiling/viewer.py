# -*- coding: utf-8 -*-
"""
   profiling.viewer
   ~~~~~~~~~~~~~~~~

   A text user interface application which inspects statistics.  To run it
   easily do:

   .. sourcecode:: console

      $ profiling view SOURCE

   ::

      viewer = StatisticsViewer()
      loop = viewer.loop()
      loop.run()

   :copyright: (c) 2014-2016, What! Studio
   :license: BSD, see LICENSE for more details.

"""
from __future__ import absolute_import

from collections import deque

import six
import urwid
from profiling.stats import make_frozen_stats_tree
from urwid import connect_signal as on

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.interface import CommandLineInterface
from prompt_toolkit.key_binding.manager import KeyBindingManager
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.containers import (Float, FloatContainer, HSplit,
                                              VSplit, Window)
from prompt_toolkit.layout.controls import (BufferControl, FillControl,
                                            TokenListControl)
from prompt_toolkit.layout.dimension import LayoutDimension as D
from prompt_toolkit.shortcuts import create_eventloop
from prompt_toolkit.token import Token

from . import sortkeys
from .stats import FlatFrozenStatistics

__all__ = ['StatisticsTable', 'StatisticsViewer', 'fmt',
           'bind_vim_keys', 'bind_game_keys']


NESTED = 0
FLAT = 1

manager = KeyBindingManager()  # Start with the `KeyBindingManager`.


@manager.registry.add_binding(Keys.ControlC, eager=True)
@manager.registry.add_binding(Keys.ControlQ, eager=True)
def _(event):
    """
    Pressing Ctrl-Q or Ctrl-C will exit the user interface.

    Setting a return value means: quit the event loop that drives the user
    interface and return this value from the `CommandLineInterface.run()` call.

    Note that Ctrl-Q does not work on all terminals. Sometimes it requires
    executing `stty -ixon`.
    """
    event.cli.set_return_value(None)

buffers = {
    DEFAULT_BUFFER: Buffer(is_multiline=True),
}


def default_buffer_changed(cli):
    """
    When the buffer on the left (DEFAULT_BUFFER) changes, update the buffer on
    the right. We just reverse the text.
    """
    pass
    # buffers['RESULT'].text = buffers[DEFAULT_BUFFER].text[::-1]


buffers[DEFAULT_BUFFER].on_text_changed += default_buffer_changed


def get_func(f):
    if isinstance(f, staticmethod):
        return f.__func__
    return f


class Formatter(object):

    def _markup(get_string, get_attr=None):
        get_string = get_func(get_string)
        get_attr = get_func(get_attr)

        @staticmethod
        def markup(*args, **kwargs):
            string = get_string(*args, **kwargs)
            if get_attr is None:
                return string
            attr = get_attr(*args, **kwargs)
            return (attr, string)
        return markup

    _numeric = {'align': 'right', 'wrap': 'clip'}

    def _make_text(get_markup, **text_kwargs):
        get_markup = get_func(get_markup)

        @staticmethod
        def make_text(*args, **kwargs):
            markup = get_markup(*args, **kwargs)
            return markup
        return make_text

    # percent

    @staticmethod
    def format_percent(ratio, denom=1, unit=False):
        # width: 4~5 (with unit)
        # examples:
        # 0.01: 1.00%
        # 0.1: 10.0%
        # 1: 100%
        try:
            ratio /= float(denom)
        except ZeroDivisionError:
            ratio = 0
        if round(ratio, 2) >= 1:
            precision = 0
        elif round(ratio, 2) >= 0.1:
            precision = 1
        else:
            precision = 2
        string = ('{:.' + str(precision) + 'f}').format(ratio * 100)
        if unit:
            return string + '%'
        else:
            return string

    @staticmethod
    def attr_ratio(ratio, denom=1, unit=False):
        try:
            ratio /= float(denom)
        except ZeroDivisionError:
            ratio = 0
        if ratio > 0.9:
            return 'danger'
        elif ratio > 0.7:
            return 'caution'
        elif ratio > 0.3:
            return 'warning'
        elif ratio > 0.1:
            return 'notice'
        elif ratio <= 0:
            return 'zero'

    markup_percent = _markup(format_percent, attr_ratio)
    make_percent_text = _make_text(markup_percent, **_numeric)

    # int

    @staticmethod
    def format_int(num, units='kMGTPEZY'):
        # width: 1~6
        # examples:
        # 0: 0
        # 1: 1
        # 10: 10
        # 100: 100
        # 1000: 1.0K
        # 10000: 10.0K
        # 100000: 100.0K
        # 1000000: 1.0M
        # -10: -11
        unit = None
        unit_iter = iter(units)
        while abs(round(num, 1)) >= 1e3:
            num /= 1e3
            try:
                unit = next(unit_iter)
            except StopIteration:
                # overflow or underflow.
                return 'o/f' if num > 0 else 'u/f'
        if unit is None:
            return '{:.0f}'.format(num)
        else:
            return '{:.1f}{}'.format(num, unit)

    @staticmethod
    def attr_int(num):
        return None if num else 'zero'

    markup_int = _markup(format_int, attr_int)
    make_int_text = _make_text(markup_int, **_numeric)

    # int or n/a

    @staticmethod
    def format_int_or_na(num):
        # width: 1~6
        # examples:
        # 0: n/a
        # 1: 1
        # 10: 10
        # 100: 100
        # 1000: 1.0K
        # 10000: 10.0K
        # 100000: 100.0K
        # 1000000: 1.0M
        # -10: -11
        if num == 0:
            return 'n/a'
        else:
            return Formatter.format_int(num)

    markup_int_or_na = _markup(format_int_or_na, attr_int)
    make_int_or_na_text = _make_text(markup_int_or_na, **_numeric)

    # time

    @staticmethod
    def format_time(sec):
        # width: 1~6 (most cases)
        # examples:
        # 0: 0
        # 0.000001: 1us
        # 0.000123: 123us
        # 0.012345: 12ms
        # 0.123456: 123ms
        # 1.234567: 1.2sec
        # 12.34567: 12.3sec
        # 123.4567: 2min3s
        # 6120: 102min
        if sec == 0:
            return '0'
        elif sec < 1e-3:
            # 1us ~ 999us
            return '{:.0f}us'.format(sec * 1e6)
        elif sec < 1:
            # 1ms ~ 999ms
            return '{:.0f}ms'.format(sec * 1e3)
        elif sec < 60:
            # 1.0sec ~ 59.9sec
            return '{:.1f}sec'.format(sec)
        elif sec < 600:
            # 1min0s ~ 9min59s
            return '{:.0f}min{:.0f}s'.format(sec // 60, sec % 60)
        else:
            return '{:.0f}min'.format(sec // 60)

    @staticmethod
    def attr_time(sec):
        if sec == 0:
            return 'zero'
        elif sec < 1e-3:
            return 'usec'
        elif sec < 1:
            return 'msec'
        elif sec < 60:
            return 'sec'
        else:
            return 'min'

    markup_time = _markup(format_time, attr_time)
    make_time_text = _make_text(markup_time, **_numeric)

    # stats

    @staticmethod
    def markup_stats(stats):
        if stats.name:
            loc = ('({0}:{1})'
                   ''.format(stats.module or stats.filename, stats.lineno))
            return [('name', stats.name), ' ', ('loc', loc)]
        else:
            return ('loc', stats.module or stats.filename)

    make_stat_text = _make_text(markup_stats, wrap='clip')

    del _markup
    del _make_text


fmt = Formatter


class StatisticsTable(urwid.WidgetWrap):

    #: The column declarations.  Define it with a list of (name, align, width,
    #: order) tuples.
    columns = [('FUNCTION', 'left', ('weight', 1), sortkeys.by_function)]

    #: The initial order.
    order = sortkeys.by_function

    #: The children statistics layout.  One of `NESTED` or `FLAT`.
    layout = NESTED

    title = None
    stats = None
    time = None

    def __init__(self, viewer):
        self._expanded_stat_hashes = set()
        self.walker = StatisticsWalker(NullStatisticsNode())
        on(self.walker, 'focus_changed', self._walker_focus_changed)
        tbody = StatisticsListBox(self.walker)
        thead = urwid.AttrMap(self.make_columns([
            urwid.Text(name, align, 'clip')
            for name, align, __, __ in self.columns
        ]), None)
        header = urwid.Columns([])
        widget = urwid.Frame(tbody, urwid.Pile([header, thead]))
        super(StatisticsTable, self).__init__(widget)
        self.viewer = viewer
        self.update_frame()

    def make_row(self, node):
        stats = node.get_value()
        return self.make_columns(self.make_cells(node, stats))

    def make_cells(self, node, stats):
        yield fmt.make_stat_text(stats)

    @classmethod
    def make_columns(cls, column_widgets):
        widget_list = []
        widths = (width for __, __, width, __ in cls.columns)
        for width, widget in zip(widths, column_widgets):
            widget_list.append(width + (widget,))
        return urwid.Columns(widget_list, 1)

    @property
    def tbody(self):
        return self._w.body

    @tbody.setter
    def tbody(self, body):
        self._w.body = body

    @property
    def thead(self):
        return self._w.header.contents[1][0]

    @thead.setter
    def thead(self, thead):
        self._w.header.contents[1] = (thead, ('pack', None))

    @property
    def header(self):
        return self._w.header.contents[0][0]

    @header.setter
    def header(self, header):
        self._w.header.contents[0] = (header, ('pack', None))

    @property
    def footer(self):
        return self._w.footer

    @footer.setter
    def footer(self, footer):
        self._w.footer = footer

    def get_focus(self):
        return self.tbody.get_focus()

    def set_focus(self, focus):
        self.tbody.set_focus(focus)

    def get_path(self):
        """Gets the path to the focused statistics. Each step is a hash of
        statistics object.
        """
        path = deque()
        __, node = self.get_focus()
        while not node.is_root():
            stats = node.get_value()
            path.appendleft(hash(stats))
            node = node.get_parent()
        return path

    def find_node(self, node, path):
        """Finds a node by the given path from the given node."""
        for hash_value in path:
            if isinstance(node, LeafStatisticsNode):
                break
            for stats in node.get_child_keys():
                if hash(stats) == hash_value:
                    node = node.get_child_node(stats)
                    break
            else:
                break
        return node

    def get_stats(self):
        return self.stats

    def set_result(self, stats, cpu_time=0.0, wall_time=0.0,
                   title=None, at=None):
        self.stats = stats
        self.cpu_time = cpu_time
        self.wall_time = wall_time
        self.title = title
        self.at = at
        self.refresh()

    def set_layout(self, layout):
        if layout == self.layout:
            return  # Ignore.
        self.layout = layout
        self.refresh()

    def sort_stats(self, order=sortkeys.by_deep_time):
        assert callable(order)
        if order == self.order:
            return  # Ignore.
        self.order = order
        self.refresh()

    def shift_order(self, delta):
        orders = [order for __, __, __, order in self.columns if order]
        x = orders.index(self.order)
        order = orders[(x + delta) % len(orders)]
        self.sort_stats(order)

    def refresh(self):
        stats = self.get_stats()
        if stats is None:
            return
        if self.layout == FLAT:
            stats = FlatFrozenStatistics.flatten(stats)
        node = StatisticsNode(stats, table=self)
        path = self.get_path()
        node = self.find_node(node, path)
        self.set_focus(node)

    def update_frame(self, focus=None):
        # Set thead attr.
        if self.viewer.paused:
            thead_attr = 'thead.paused'
        elif not self.viewer.active:
            thead_attr = 'thead.inactive'
        else:
            thead_attr = 'thead'
        self.thead.set_attr_map({None: thead_attr})
        # Set sorting column in thead attr.
        for x, (__, __, __, order) in enumerate(self.columns):
            attr = thead_attr + '.sorted' if order is self.order else None
            widget = self.thead.base_widget.contents[x][0]
            text, __ = widget.get_text()
            widget.set_text((attr, text))
        if self.viewer.paused:
            return
        # Update header.
        stats = self.get_stats()
        if stats is None:
            return
        title = self.title
        time = self.time
        if title or time:
            if time is not None:
                time_string = '{:%H:%M:%S}'.format(time)
            if title and time:
                markup = [('weak', title), ' ', time_string]
            elif title:
                markup = title
            else:
                markup = time_string
            meta_info = urwid.Text(markup, align='right')
        else:
            meta_info = None
        fraction_string = '({0}/{1})'.format(
            fmt.format_time(self.cpu_time),
            fmt.format_time(self.wall_time))
        try:
            cpu_usage = self.cpu_time / self.wall_time
        except ZeroDivisionError:
            cpu_usage = 0.0
        cpu_info = urwid.Text([
            'CPU ', fmt.markup_percent(cpu_usage, unit=True),
            ' ', ('weak', fraction_string)])
        # Set header columns.
        col_opts = ('weight', 1, False)
        self.header.contents = \
            [(w, col_opts) for w in [cpu_info, meta_info] if w]

    def focus_hotspot(self, size):
        widget, __ = self.tbody.get_focus()
        while widget:
            node = widget.get_node()
            widget.expand()
            widget = widget.first_child()
        self.tbody.change_focus(size, node)

    def defocus(self):
        __, node = self.get_focus()
        self.set_focus(node.get_root())

    def keypress(self, size, key):
        command = self._command_map[key]
        if key == ']':
            self.shift_order(+1)
            return True
        elif key == '[':
            self.shift_order(-1)
            return True
        elif key == '>':
            self.focus_hotspot(size)
            return True
        elif key == '\\':
            layout = {FLAT: NESTED, NESTED: FLAT}[self.layout]
            self.set_layout(layout)
            return True
        command = self._command_map[key]
        if command == 'menu':
            # key: ESC.
            self.defocus()
            return True
        elif command == urwid.CURSOR_RIGHT:
            if self.layout == FLAT:
                return True  # Ignore.
            widget, node = self.tbody.get_focus()
            if widget.expanded:
                heavy_widget = widget.first_child()
                if heavy_widget is not None:
                    heavy_node = heavy_widget.get_node()
                    self.tbody.change_focus(size, heavy_node)
                return True
        elif command == urwid.CURSOR_LEFT:
            if self.layout == FLAT:
                return True  # Ignore.
            widget, node = self.tbody.get_focus()
            if not widget.expanded:
                parent_node = node.get_parent()
                if not parent_node.is_root():
                    self.tbody.change_focus(size, parent_node)
                return True
        elif command == urwid.ACTIVATE:
            # key: Enter or Space.
            if self.viewer.paused:
                self.viewer.resume()
            else:
                self.viewer.pause()
            return True
        return super(StatisticsTable, self).keypress(size, key)

    # Signal handlers.

    def _walker_focus_changed(self, focus):
        self.update_frame(focus)

    def _widget_expanded(self, widget):
        stats = widget.get_node().get_value()
        self._expanded_stat_hashes.add(hash(stats))

    def _widget_collapsed(self, widget):
        stats = widget.get_node().get_value()
        self._expanded_stat_hashes.discard(hash(stats))


class StatisticsViewer(object):

    weak_color = 'light green'
    palette = [
        ('weak', weak_color, ''),
        ('focus', 'standout', '', 'standout'),
        # ui
        ('thead', 'dark cyan, standout', '', 'standout'),
        ('thead.paused', 'dark red, standout', '', 'standout'),
        ('thead.inactive', 'brown, standout', '', 'standout'),
        ('mark', 'dark magenta', ''),
        # risk
        ('danger', 'dark red', '', 'blink'),
        ('caution', 'light red', '', 'blink'),
        ('warning', 'brown', '', 'blink'),
        ('notice', 'dark green', '', 'blink'),
        # clock
        ('min', 'dark red', ''),
        ('sec', 'brown', ''),
        ('msec', '', ''),
        ('usec', weak_color, ''),
        # etc
        ('zero', weak_color, ''),
        ('name', 'bold', ''),
        ('loc', 'dark blue', ''),
    ]
    # add thead.*.sorted palette entries
    for entry in palette[:]:
        attr = entry[0]
        if attr is not None and attr.startswith('thead'):
            fg, bg, mono = entry[1:4]
            palette.append((attr + '.sorted', fg + ', underline',
                            bg, mono + ', underline'))

    focus_map = {None: 'focus'}
    focus_map.update((x[0], 'focus') for x in palette)

    #: Whether the viewer is active.
    active = False

    #: Whether the viewer is paused.
    paused = False

    def unhandled_input(self, key):
        if key in ('q', 'Q'):
            raise self.eventloop.close()

    def __init__(self):
        self._fc = FloatContainer(
            content=VSplit([
                Window(content=BufferControl(buffer_name=DEFAULT_BUFFER))]),
            floats=[
                Float(
                    content=Window(
                        height=D.exact(1),
                        content=FillControl('-', token=Token.Line)))
            ]
        )

        self.layout = HSplit([
            # The titlebar.
            Window(
                height=D.exact(1),
                content=TokenListControl(
                    self.get_titlebar_tokens, align_center=True)
            ),
            Window(
                height=D.exact(1),
                content=FillControl('-', token=Token.Line)),
            self._fc,
        ])

        self.widget = Application(
            layout=self.layout,
            buffers=buffers,
            key_bindings_registry=manager.registry,

            # Let's add mouse support!
            mouse_support=True,

            # Using an alternate screen buffer means as much as: "run full
            # screen". It switches the terminal to an alternate screen.
            use_alternate_screen=False)

        self.eventloop = create_eventloop()

    def loop(self, *args, **kwargs):
        kwargs.setdefault('unhandled_input', self.unhandled_input)
        self.cli = CommandLineInterface(application=self.widget,
                                        eventloop=self.eventloop)

        return self.cli

    def set_profiler_class(self, profiler_class):
        pass
        # table_class = profiler_class.table_class
        # # NOTE: Don't use isinstance() at the below line.
        # if type(self.table) is table_class:
        #     return
        # self.table = table_class(self)
        # self.widget.original_widget = self.table

    def set_result(self, stats, cpu_time=0.0, wall_time=0.0,
                   title=None, at=None):
        self._final_result = (stats, cpu_time, wall_time, title, at)
        if not self.paused:
            self.update_result()

    def get_result(self):
        try:
            if self.paused:
                result = self._paused_result
            else:
                result = self._final_result
        except AttributeError:
            return
        return result

    def update_result(self):
        """Updates the result on the table."""
        result = self.get_result()
        if result is None:
            return

        self.update()
        self.cli.request_redraw()

    def update(self):

        def create_layout_from_stats():
            result = self.get_result()
            stats, cpu_time, wall_time, title, time = result
            if result is None:
                return VSplit([Window(
                    height=D.exact(1),
                    content=FillControl('-', token=Token.Line))])
            else:
                stats, cpu_time, wall_time, title, time = result
                panels = []

                for _stats in make_frozen_stats_tree(stats):
                    name, filename, lineno, module, own_hits, deep_time = _stats[1]
                    if name == 'run':
                        continue
                    else:
                        panels.append(Window(
                            height=D.exact(1),
                            content=TokenListControl(
                                lambda x: [(Token.Title, ' | '.join(six.text_type(s) for s in _stats[1]))], align_center=True)
                        ))
                return HSplit(panels)
        layout = create_layout_from_stats()
        self._fc.content = layout

    def get_titlebar_tokens(self, cli):
        result = self.get_result()
        if result is None:
            return [
                (Token.Title, 'waiting'),
            ]

        stats, cpu_time, wall_time, title, time = result
        if title or time:
            if time is not None:
                time_string = '{:%H:%M:%S}'.format(time)
            if title and time:
                markup = '%s %s ' % (title, time_string)
            elif title:
                markup = title
            else:
                markup = time_string
            meta_info = markup
        else:
            meta_info = None
        fraction_string = '({0}/{1})'.format(
            fmt.format_time(cpu_time),
            fmt.format_time(wall_time))
        try:
            cpu_usage = cpu_time / wall_time
        except ZeroDivisionError:
            cpu_usage = 0.0
        cpu_info = ' CPU %s %s' % (
            fmt.markup_percent(cpu_usage, unit=True),
            fraction_string)
        return [
            (Token.Title, meta_info),
            (Token.Title, cpu_info),
        ]

    def activate(self):
        self.active = True
        # self.table.update_frame()

    def inactivate(self):
        self.active = False
        # self.table.update_frame()

    def pause(self):
        self.paused = True
        try:
            self._paused_result = self._final_result
        except AttributeError:
            pass
        # self.table.update_frame()

    def resume(self):
        self.paused = False
        try:
            del self._paused_result
        except AttributeError:
            pass
        self.update_result()


def bind_vim_keys(urwid=urwid):
    urwid.command_map['h'] = urwid.command_map['left']
    urwid.command_map['j'] = urwid.command_map['down']
    urwid.command_map['k'] = urwid.command_map['up']
    urwid.command_map['l'] = urwid.command_map['right']


def bind_game_keys(urwid=urwid):
    urwid.command_map['a'] = urwid.command_map['left']
    urwid.command_map['s'] = urwid.command_map['down']
    urwid.command_map['w'] = urwid.command_map['up']
    urwid.command_map['d'] = urwid.command_map['right']

