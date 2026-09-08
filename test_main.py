import unittest
from unittest.mock import Mock, patch

import requests
import main


class SolverTests(unittest.TestCase):
    def test_wikipedia_links(self):
        # wikipedia uses both link styles, and may escape article names
        html = '''<a href="/wiki/GitHub">self</a>
        <a href="./Microsoft">Microsoft</a>
        <a href="/wiki/Warsaw#History">Warsaw</a>
        <a href="./Caf%C3%A9">Café</a>
        <a href="./File:Logo.svg">file</a>
        <a href="/wiki/">empty</a>
        <a href="https://example.com/wiki/Fake">outside</a>'''
        links = main.extract_links(html, 'GitHub')
        self.assertEqual(set(links) - {'GitHub'}, {'Microsoft', 'Warsaw', 'Café'})
        # a target near the end of a long page must still be found
        html = ''.join(f'<a href="./Page_{i}">link</a>' for i in range(150)) + html
        self.assertIn('Warsaw', main.extract_links(html, 'GitHub'))
        self.assertEqual(main.page_name('https://en.wikipedia.org/wiki/Caf%C3%A9#History'), 'Café')
        self.assertEqual(main.page_url('C++'), 'https://en.wikipedia.org/wiki/C%2B%2B')

    def test_fetch_errors_are_not_dead_ends(self):
        response = Mock()
        response.raise_for_status.side_effect = requests.HTTPError('403 forbidden')
        with patch.object(main.session, 'get', return_value=response):
            with self.assertRaises(requests.HTTPError):
                main.get_links('GitHub')

    def test_route_and_no_revisits(self):
        # this small graph checks real ranking, direct hits, and visited links
        model = Mock()
        model.encode.side_effect = lambda texts, **kw: [[float(t != 'Dead end'), 1.] for t in texts]
        graph = {'GitHub': {'Dead_end': 'Dead end', 'Microsoft': 'Microsoft'},
                 'Microsoft': {'GitHub': 'GitHub', 'Warsaw': 'Warsaw'}, 'Warsaw': {}, 'Dead_end': {}}
        with patch.object(main, 'get_links', side_effect=graph.__getitem__):
            self.assertEqual(main.solve('GitHub', 'Warsaw', model), ['GitHub', 'Microsoft', 'Warsaw'])
            with self.assertRaisesRegex(RuntimeError, 'dead end'):
                main.solve('GitHub', 'Nowhere', model)
            with self.assertRaisesRegex(RuntimeError, 'step limit'):
                main.solve('GitHub', 'Warsaw', model, max_steps=1)
        self.assertEqual(main.solve('Warsaw', 'Warsaw', model), ['Warsaw'])


if __name__ == '__main__':
    unittest.main()
