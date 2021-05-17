from tools_parties import extractFromName

import unittest


class TestSum(unittest.TestCase):
    def test_extractFromName(self):

        testCase = "Alessia Maria Mosca (S&amp;D)"
        result = "S&D"
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))

        testCase = "Fulvio Martusciello (PPE)"
        result = "PPE"
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))

        testCase = "Dominique Bilde (ENF)"
        result = "ENF"
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))

        testCase = "Xabier Benito Ziluaga (GUE/NGL)"
        result = "GUE/NGL"
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))

        testCase = "Xabier Benito Ziluaga GUE/NGL"
        result = -1
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))

        testCase = ""
        result = -1
        self.assertEqual(extractFromName(
            testCase), result, "From Name '{name}' the party '{party}' should be extracted.".format(name=testCase, party=result))


if __name__ == '__main__':
    unittest.main()
