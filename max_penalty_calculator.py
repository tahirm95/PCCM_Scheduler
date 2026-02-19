#!/usr/bin/env python3
"""Calculate maximum penalty for the scheduler objective.

Modes
-----
1) exact: maximize the full-weight model objective via CP-SAT.
2) bound: compute a fast theoretical upper bound from objective variable domains.

This script is intentionally standalone (does not import fellowship_scheduler_2026_02_17.py).
It loads and executes that file's definitions (excluding its bottom run block) in an isolated namespace.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple

from ortools.sat.python import cp_model

SCHEDULER_FILE = Path(__file__).with_name("fellowship_scheduler_2026_02_17.py")
# Embedded fallback snapshot so this tool can run even when the scheduler file is unavailable.
_EMBEDDED_SCHEDULER_GZIP_B64 = """H4sIALBzlmkC/+19a3PbOrLgd/0KjlKnRjqRH6JsJ0cTu1ax5cQzfuTaTjK5XhWLliibE1nSiFQcX5f/+6IbD+JFEpKc5Ozs3pp7IhNAo9EvNB7dePG/0jgdRV4nTaO7aeq9anpr3od3X9a2vLeTcDZIKpUX3v5kFF6vzaJw8OAdRqPR5D65jafeRf82GsxH0cyrnZ2vXU4mo8Tb/7B20bmsV16QZh+T8Cby4jFt3yZfPO8vU9Jw7d/ka5KGo5E3maXYbhqOB2GCVX6bzcfeUHQTJLyboA9w1qcPCL0z+Nc8Sb30NvL6k/EwvpnPwjSejL0k6uO/YeKNo2gQDbzaIEyjpOHdR/HNbUp+zKJ/z6MkTerrlcpwNrnzoEIa3xFs76YEJfy74cGXQTRKQ1ppGqa3o/ia1/lA/qyw31CTVkofpvH4htc5iPtpwzuOE/Lfy/l0xCqxUa8nYUqGk94SbFmD/jS4m5A+OWBKFxjLdACsWMv/P2SURIiS2heXnfPL4KBz2fV2cbw1f9PfaXjkf/4fdc974Z1MSN8PldOPJ8HnbvcfF6Tetl+pdC4v8G+p2auGt93wmq9IM6nh/vvuxaVelYBvbpL/J12Iih8+Hp8Eb8865wcXOdW3pdpkXG8nc/JzFkeJN08Ig8P+bJIkhPPfolk48mZEWpLKh/Nu8OGMYPD27OPpQef8i4pws+FtScP0am/Pz07333vTWbQxnRDJSqajOP0b/O298ZobWxvQCkv2dtnf9cph5/g46J4eaCj7hIivkRrs/7JuDkHs43F/NB8Q9NPbOCFyGX2tVy4+nB+dvguQL1ZUDVAX0xnIGtGkWaqAIiQ6P7/c2N//6N3H48HkvkL+DMifudBFBXUsr6g87CChALY3GXo7G36TDB566Rydf+58YQR/QQ1H25uMRw/eYI7ITWZeOEyJjQBF5RD+2PBfb/g7rMV22wvHD1SHKMCAfN4ycCWY/IF0hZ4vPxxfLtxvk+AN/ULj3D6ahCA+9nF6uHAPrY3mNtDm9NAOnxC0RbrYRvifwj41WdejsP91Mk+J0EKj07PgU2e/E7ztHp6da+pJAIDcKfW4dj5y1Wx4me41NGY2t+r6J79ZfwKI2wJi95+X5x2A6NYW+DG5uSHzyJBQ5eKT9y1OQi/63o+mOD6gYqV72nl73A0uPgWfji6gi/3uh8ujs1PSzeVsHmlA6Bw0jGdE35hmIo2nYYzUZyAJgfa7weHROeg5Vgs+EBkiisTBWqDehd8JJyMOd0p4CLAFOCQug3bS+Wdwdtq1QuNwKFQyTjBkXi2JyJQ0TyfDIdLKu47IVFZXoaPJY7CDD+cu8JmROzna/9h8Cf/13bo6Obo4KO1KIvcaToTQgUpkBEYpjSDPPx7bYR0cUCS9ayLVa+lkDf4FO/otGuPExCEeHIDA7Z+dXnT3A9okD7cdb/82joYJUPn8gjCNKeHZ6dsDg3k7wf77o+7hRQA1A6xilbHOO288Sb37yewrwLqP01uVlxuI1AYl9nVE2kTEtjPjxTvsvINRIHW6nfPjL1Jfh/F3MjsBeklG5QogZJnn6MRLNbt73N2/zPRa0//NuqzgqH4XwC5uRirwV/D2uLP/j7OPl4b92EHj85k6Q4hXMhmm4EUl6SyMx+RjLZ0TuRMeVL3yuXv07v0lIlOBSaj6LewHs3D8tVlte9ubm5sN9bNPPje3jc8t8rmV1R4SZQ5uiZsJhTcRKfTVsru4P98RhQQgK8Q22wHxFClA+XOLfd7iX6fz0V0AdiOY3jxsYQkvAvg5RTDTB+NhQFzAeH6H3Wgl0Bv5vsO/jxOANp6MgZJRHwnDivp5BbN4GNzfRmMyxkFChjtGGvBC4voG4QycnYBQMAQSqKjjBBSg2hKMSPlrgeTNw05AdC64Rjc+AFgwiG2FvtPbdJzRXpQRVIMkHocxkgW5EEThbPSgUBvLoIv4jvciiOUXVNLphsOMvof9tCWzrT+Kx3E/IIOPZsN4NArSOJo1FaraavgKlaxVWoosWatsKbJorbKtEMNaZYdLFNHRt52LbpCpEfu13p9MH2qokNTMjUIy4b1mNoPQbRjNonE/8mpgyzZgcq7zhQwxLenknhDTi8aDCtq9487F5esAuiJdbG1vyl8PuvsdME4t5ev+x/NPUPmqiZ4g/Z+P/yOOylYP/aDJeO0CxIEaMGIe5lOcbh5QKOqV07PT4OLotHNEzeBp56SLpqL6/qDa8Kqf/lF9EgNkdlwHkZltqXkHm1+ekOaXJ9Sa22wiemwtyTVh31qbdei3c+DdwVLxOtJnD+K5JdR3WyOqmRLbB19xtVjpHGQd4jLJqVcfPaKGZTGAZvpoTPR1TJYoJ0RR+jH4IKgTwnTTzrPJIlmHaYQuhr0RWUcSpt8SkfD64RjmLjomOo7ZJEV3MpGGF888qnU4yvXKkVho8ekhGxsV5Lf/JEIrof8aF2tMyv9+YBa+5oXvzUJ/mxceWApf8cLTE7UQSNvc5KVEfNogg2RF7oEwkuV2iBP20QkbHNEveWBn+//oEu6dXXbAv6SiBNM5CBPO6PyHDz+oo0dEDGQQ3F/iJB2dnR9dfglOPh7D9Pm40wZlIOsUwkqyCmg+ASK3RAeJ57jLf5DF2GQWp7hAPZ+P0YtKCNWnXm2CbnA4Isusj6fB5dEJcYOJXH0gjclyMIkq2afg+OjkiLjxxCU6Oz0A3K98YhK9LWLZvVfErHp/bG725Abn3Q/dDtqUViUTlsph9/j47DMCqNBlI/U+4HcN2AHLvgb5eXoifv49+0rEgP98n30lrMCfEsRtBvHD+yqs06HW0aX4+aUjfnb+W/y8+JR9fUd/ShB3GMSL4yo4KwjmQPy8PBE///5W/OxIFS7pzwqxXbIMUDIIQUAXC358eP/l4uhMkgSrlJx3Di7w36PDKpNL8LLg0yn/AfYC/qWLWDRdZKGJJef4zz5rcIhWkdhy+BfcQQ7yw/vL0yog/p6IsiK3hbhZoCOkp8pJ9+Do4wlC4rioaD9Vup2LL1jBQhFl1LYBKoMgKrR/fHR6tE/816N3R7DakxlQy3qwwa3zxmTJ/K5L1ufdc+4l76BOMfPmRaP4Jr6OR0TXvOsHEJlKB2Sd6Dz5DUOZtb3HLdyT2nlCWzqDbUiBy5Nc/wrJ14NmWF8tA5pimV6AKOe1wuFh4ZZeQgZsLUBmWEtOpSIyJq2UktTaEHliLWH8y4OJsiONTd6q4FunHpmmcLYKvdoNcVGnZGISZbiSAn+ffKS2L6kTIN3v06gPMxhhyV2Ywo7wI275VqmxImb+Cv/22HfUCQZAKsQKVdz24vPGF/J/jZOTxsEBsQHVUTS+ScHdhf0eYrmrgAv8+dRYFYS/OoiWBKLHfolP6+vr+AsKniqgXiCyxMj/10ey4MvmaWbOyUrtfuL5a7g3cj2a9L8mbW8IW4wvvQS3BxsGI7JpnppEOhCZzNLGpT5EsQ/3ShmirxK5FAA4RbibZAXhO4HYpHOIFUJLgtCTfi8yTrpX93r5cVIAvr/8MAmErQVH2ZNdtV/I4CLuOPG3CIVnYi8xb61V2fv6l3D3/UrcLdY+J/6C/761Kn9xqfQjGewXqZ8Th7dXGydlMSygFufxwWo8pgvglXQYlmE7LoNfgUWF5HFlUWsRJHvyqvNX6pA7fZfk8vOZSX/rp7Do+ZUINw/+b54Gi4XkOTic7VGtwuJCDjnx2C+YTHNYLO9DbOxkCxC8ZdEfReGYOL77F5+oLHRWMqivV3IWSgD4qwJoPZs19ndWlQN/Z1Vd93eW0fXOu1UNer5D8lNWNTaPsvkjFjXNvC6aq+Loqun5btkzjXJHN92LjnKnwPb7rpNecyk5/u8V7ZQ2WzcXN1TNV8szmEHY/JFizK+R+SsN1JydmysvYp5fX5v0HGUFUfYLNN5JlMEH+dHD3C6SmeW8zObKM0/TTWOPLn/1Us13tFaryWGzuZJv0KTHhSvYVFMXXLfD3q7KodVNTb5sOa8E/vjRRrWoj2ce6LO6qc2VlyvNJab3H28SW6uOchWhK6NTkcJ9eL+SwhXNzE3Xmfn1n3tuty86mkuY/hVHalm3NH/EXuxqy45CP2i5HYbmn2/ZYa4aljBM/tYyKntxvLTKli13fs6Ka6do+f183vjWyk5Qc1UnqPl6GSfo4tMqDDangtbKhymtxbdHmtt5EP48PpC5Z7uMs+e3ltHiy5MVdw9WWYXZdil//NB/wY536YahqyZv/dn3wUyVay5ulhdzpIQoX644Ia1+bLXi4W8hhB8nysss2vL30vw/z4ads6+ri9KXzsrr/9bKc0Jrc2U3/dUPtot/rLweeZaF189Yjfj+ypuqq9nFooOYP8tZz7Pskyx2FiBUdtUbbP/5gvzn2UH44T7bsxyEmCunJbYQfsJByCoHeGUO15Ir6XKlrTyVZ6h4H40g3BrujZfUxbgrDJ/H4A0pXcXLLDFHDePkduM63vKP4ZY/RjPWRNKKeo9COjo96P4TI7japB7WbngDaBCN53fRDIYudVmn4UBb0LWoXgsa3vTmoa62YoE25OsQSr3dXQibg3D65RpvY+Od5Rrv9CoHB8HRAQx1HH1PaxmIcXgXNbygAAbUACBw81AP6lPRobDcaOGF4wEFTaraIgV7UvDfxdL97Cj96OGEPQggXJUsZGVdr3zprAyH+KJ1jE1ikCBeRQCaqQBE0ApIZGUQDb37ePC9NqDaWvfW9kj9tI06SPoYYEQ7gZAJfVvo6iyMk8j7FI7mUXc2m8xqw+oBxGA+Dp68OMGWIU8uEmNUoceT76xXadzALErnMxn61aDH8QJVDGIWT1xDoxJgopMk/ha1WU6daDzQP+IgIEfOFRlJry33c+WiqjBurTfvzS5pQf6jdKdiSuP5FUqWIpEWY0HK33gaQSbjYDKjgdrP3dfertpZQKlAAARoyuXu4AftqVqtntPOGKsx/hWbYt+QiwVqr5OKMmoDb00yvKRhsjtYh7bkZ61OBBpCfHnyHJQBGsdKI9cwHQ8P3lUYYOTpqVfwT7WyTEZLCwh8DEejDZYSR+r+G4tcSmiuHg7VWbJQjHianx7P0rMEGMItOcUPRlZD3CaYLczmQBOcxONB3If4Y5bLhdkI1Hr+rS7lVpLLs691I6mSXE8vq2c5KeRq4mNdiYbGcVMrROddHHoymaXRoGaGTZMZ2BJ1LOK4UcTANLZpp1ZBrlPrA51RO4td5gUzr8dpdJfU6plRZNYNgoipl6KmRxJE51mQVNkTNk3JomTJhFOXciKFEIYrsn/UeF4RiHWXkxzlS7mRCqnO0x4VgM4yG+UD1rIf1WmmowKgPJlRPkgl3VGdWYI1miSLQwTmYVYZCjpLN1JH24A5ZZbRTmM/tcezKWCmnzXIoCPnNpHy5RCssiljMgQaMGxZshfMLof1gUoXUcpC7UchJn6aENs4Cr/zJHSsCyZUyXrFTPWz//Hy7PBQTQiFp5IVM1dPSV0Frt2y5nRv6ywfgAUnKufvyOjJkGOabo+RlXrkHv0+etCz5hDhz9LlGAPM1qJyNStqBhjE6e9z6JLnUpBEGW6HfxMxrMShqVf+/vH4yxICt343Gae34Me9QtM9WIcUFvA3DKBXvgQ6wfx+xbVwSr+ex6MBTQdYq9O5G/8g6PI0gev70xNajsX4HyAoWbUJ1xM/fiYfxZoIv5yzapmDmYF44R1E/TgBWn0LZ0nb+341bHj3xDntYfl3MNxPwiQPs3XXYT1zN6HoPiv6LBXx4llWfK4VYz9XNdZvHcKR6ZBPo/u3k8noUwj+6/dg+Dh8Cu4f75+C2ePsqSqNAv8DhPwGVdvgJxNg7N8ZnXKInEr9Mk9HdMtc9Cus3Kv3NNgQ0RyilION4ek3MJ8YzV+ZpRZbjVR04J3BoJbM72oKVQwq1kEUmwozu1LAvGQJM6TKFpiL8ZSvwYRkmYxlyza2VpGDzympzRYqGZCjlAZQH8e8qYz5pDR3GgSpx8JuUZstXIbCDGoWcuQY1Dw2GsMDvmaDYjke6pnIAPy699KlnW+0M1oRj7apfCwgnUhr504vPbldHr3MGaxI7KVhYlIMgzpvUO7ZIMjsJE/6evI2WmnNOyKTPhtonPCxNrC1nGAHvua3QW6ZjZAZxa18e6umTlY9x1xGp/AmIF7zStsRnXfVuskhfQJWWfOCCkVCYKwJwxd9R19qMlaqTm+2wFeELGx2G06GAHUwTRsY8qoqsIT3opwAcBGFfCnKIEES1/r6GXFwu2PSuh8dDWsSqk4w0Orkw1g/naS1eh4kGATlnjKSEphMwKVksOBSo3lzZARkc2sWcgJrFLGCg3CyWiWEpLAKuIEV3KAU0A4rLMYQbkzLoGoskYr8UjL7ZWT27WT2lyGzX0Zm3w1KCUH8xcncLCezr/qnxJqtXYeQ/HkQJ7h2Dcewgi9xtIjdwuRGaINhFUgzpcE+jPDTrK6GvB2TpUgyHA/aB6YcxVzBbBt1MsZEZZhlVO6mxJ0iVlpaf1zd97y/7HpiV8Z0kmwOEk0DlIMo26vIMsAi0jStMUvS/DcaXQvbgpihWcKNzSZXw95Vs4d7/aY3WDA6BuVeuIH6toi7F8gzGvFhFvaj7ZEs0Aum0HLqQ9kyWaCH00M7fFvqwGs1dWCbb6zI3kQyJQu5YRwNNGiQs24t7M9hSeCSN1BhK65tAuo/yEKw2dOlV65q37DLIdD1PQGdV/1KAtszmiprkIK0gI5cuc5dZFBK8p1Eru4s4bq7hguZUTYfHZUbcpfJiJU2gIRmluUSzVRMBQi3TVRDCrLczjeOyu6xYnIyh9wti7O5nDCTOavLiaF63Je7hMhBlaaeU1G9PDEShmKWZK5LGoVwx0jJT5pHKnoG2WBHePWc7i35SvO6Dwere/8H1WxznXr+Bw5DoT0zISwYxG34LfIitk3iy6nck4p9jSdDZlkP66YaYXe+6O5LR6YZXeHTxKgshzyerdR2Nvw/cBfSHBA91zUFRPICTQnRRkc3HCdDlv8ctge1j37OoIVkaG60dcxuIPwyEDnD2MowPj106AunLHtHW5KNEZqiJzjPJOyeHzwN6lYV3yrR7XuR1tOyCfaBLJ4m43D2wFPuwijbuOVPTRPTNcPw7RQYPtshm64QEg7czcy63UYhKUZg23tZhofmndr7Z0h8JgiMYCMQHku5ibxBdEdwSCols5XzZgx3umVRKwRgLBsVCE1HCP5KEJhnlwtAQJjePpCFdUCJRuR1k07hpJZyYBuNkkjaY8snHl9H2HtWepOwgGXM5angYdtr6jvODfb8hrm0wZzvz4f/5WkB9qInBffzKInCWf9Wwj9MvVEEqc9b/Oknde+8EAkm7hYkyHq3JXV9PZuM+7dBfzIfp+a6WplsVEAWHBRYe4qU6fuh6lMijnucJjbyLmcJPm+yuZGS/IK4WaMoJSbmXlN/+sJNOY3Pc/TjjZuCUb9z+fZillmuuVgfroAB5uFdBcLKACB78ErtIS9xCQC+YV1wOsksUDRbY0ZHbDyi9CWOIMT5XDgYwNMXwV34XZzTZedzDQ/K6EVPmgWe1JP+luaolOAxMhSbrd5MF0VWWNGHfFckT1VpP0TpRSsVFsfPHRahv2hV188jsUpGMk7tAL8nbXxl7gqfl7tC2hGi9RrZGfHROP0UznrZme0yB3/ShU51JBo2eDbJfALoUuFtQ6RffwWRLOVwmGtgA8TToG9DogA3SH4+JN8dEnUVbIBYhvVWw9tygMOnfiuVeAb2bXyupxQWs002UDw7vd/wfAdI3FDaQIm073CN3gEWNTc2SCz/Ozwy4jK6w5yhHbqPCxPVW6GwFPZbblzj07INksjRv+PGNbZWsYHimfIdR5fP/sW4TzdubHDoKwXOnD/KYRlN+L+laFo0ki6//7mMiyPZHIyLI6RS47LlZqScjEvTUf3+nMblPIdQ9DUMVygXecrM3rpovsLApv9AI+XIfbrCs8sRKcHpTlHmhLhtxPvjYRnPoM+bbqN20Ofms+lz85n02RGOkz5vusF6Hs3Zz7MJ+4tYhFL9gzuhLU1O4WAJZIvuQponJYvrp6lXKMfPAPn/Dc1XznjojZQJWY/A2Q5b4PDd+Ra8m7tVL9xlhecG7Us5sVf40svZ/stZ4WXrLQk63R1aaNF7ARmm4T0+/R6lS+PTibfne/QRyHkKd87FvbIF7oZ6a0SiHHbC2b5sRinys5nz3c++vxGbtrlYI4Nr0mXK+rOOwdA6V/6/zG3Jhm62lgpKIfh5EHwJApAv/2oloaVMSFgGgT4Qr3gJCjZdpIBt+RgjZt/VjaN8KLjxZALBz8q9S22Epz9sYKdWjE4NhA4O2t54or5FTI/nskNIacPWfJN4BVRpKK5NckWJJplvjCPzNn0SlJ8jcjuKr7aJmxtF5tSmew2v+Iq5tHlWz66YQ5/GqdLk+l9RH3gdpNHsLoFYit6CNlV9fdibRdPJLMW5oz8ZjQh0jEJAPCC4cC2ZjIho0Wp1x27gieOANhFRZ/xpYqQhxOb3srj27NVgvUC8GawXwLu2LdvHbQO2/rawtZS9k6uUmW8MK8X9okLbW8NqBfO9YXPkxpPDKu72Z4fNOiDUr7HY2lXRC8Q6MP19YXh51zo08cawUiQeobYW4AvU1hI43TLor70CLJU/Ca3GR33pvQ5xCrNHHKPrB28ajcNR/D/8niB7SY+fzLew0XrZSS2ONeeMK/eoTG1NKWW/vSpV0C+vZkZExgG8LO1epwSjpPVfilvrV001a7QeTglJBzX22vKVIgc973fPiohkK7QWHF4NLKPUVJr+1KeS25yl1Gtqbvj8Sii9lvIqu2zBzj1v4ynCQmmngk+r7OL7vzXKMaK9tVdgxdnYpdqjaMxCu3QQRlW4QJu9RX1VoHU9Pk9qHcF8lUme0iF9Xp5MB1qHQHYVSsVccNkhqavAMSEG2KqaPqLfvZqG5poX170NrVvthrKWQUSrmyU8qVjxC64fkMgQzRj309r/xFOD/g3LsLjoCI1UM2RYpnEdqjaPk+8EB0eXWWlKMSKNcwd3da9et8zRNgbod0RGu9ouq1aBuAlFo/hCEX/2vL7oOSVeamQXz4iPMe+n81lEYzcn8+tRBLkPhkNic51BXs7Ai6SAqR5L99q84Si8ScCaZ5F/OFuyE/pMUB7vYXowTfmTejPTgE/qR2H/lsc4M49QHgxcEoSZI+RPI+lu7uHROeRPoBcTPnSOIC9B2+XSVZywuZ8bpaveone+r+3zCsqAQikR2lk1L7eqeHBxuTYr2mhP5FhrYTR7AWjuEUsdJywvQ8IzWeTfs70Ge2i5w1HcB1md4Nvh5uVI46bwnrdpv7ZMPKBvwAze+dS4PDnNmHJf75XcfQavAUDaozLyaXY0pBQj/8vmOGlYDXyiK55FuupRcZWFVZGYnHs71qsyOX5NFp6zvcPEwQ2UdAOv6I61iiNsotipZm+tYGUJ0rGR/IXX1ayAco8TQjgNBhSs+VSNstxePMPYELzwIS7y8stasK7w4hQYH+YZxjK/FbZ17cpatFTNaWLb2lj9+lfxlasmPLpcgJK6V8pT5dAtV5yP8G95+wRfdGYNzuejyJtPs0Q+EFXaYfdItxLrTV7ZougxsOtFV/xhRZWzBWveyRapfTJC4RrWHYDI92OEV0AeETYKZjruonHa1m8rWxYR0hDUm8gMMOwkMMhzwijg9cAGRhqJujWawaF88mrqNql2mQcBYUWa3kabPC3EsISKvCRC9sb7jNJSwz/rRW2KpttJJqMs3U7BdGsgL6+EruuOwSeC8wUmrhgI2y5zBaSaimskckAQBpoZQzJMHjOy4wT8RSYqclRf/3aSRBSQpKlOzMSFP8x7RAYoWsl9Hl6Y3wyv6N57WBGlAP404q042PaC/HhDlZQ3Vy1Vh9/N3YZl6o5pb0Jviz5AL0t/5nBH4VewJ2tK8D1oJy0LqL8slAF3j1AutzJFeRwKV1lYq6ciA8Zg48Zhjp6ddwvUrKWqWate0AT7y9EyaTBF+sUVca6HuJO+v+o282s2OW3VHXwSgLoHNlBTlMSx7RvLHkxiC/LVCM9thKUfk8lXQ2V3JZFA34YJk5VyIkff4yRNkNZV5VKmhFeOO3ESfu/+ex5CmpRa1mVDbqndc8jxC2R8lcg8Xej5gLMGkuIZ3rjSFO+2q6evl7ewKX4XxmMwLNQJw6wdkAtJ6ChoIutLVmGu4+n9hIHzc3UajVOBP4HhW0P1RileXlf9C9hDL5kKC1XUPhPmq2jRujMTIYpW4Vyo4l6w9vxzTIJwu05Fucif1qqinBlrGV0arwjLe7JFUPZeTg892F/HUIO2JHT9yd10FKXRIBM/WLTQ0zmWLc7nYgZpKfXTxJwdsoJQW9BHx8lRCsXW6xOdSOPxPFLzPsAaPiBoBmwPDqiZe548te7MTfXbDMqq3XKubHN7FTRsGR+0QSrnjVv0dH8yG1D/RGIc5Quyb9Lvz2cyZxDp9Z/IHSU5xyIMsuUumepRlwrZiymdn7/ExNaiG0htXDCuQXoFRmqiD3cTpg4iqp8qDQ3Op4trbCyW1i/YnQyhOnWRxReCfbNAfj0+iW4GYvaq84/HXfdsX7jOxW1bzYdz2EhkeQjCxBvAKLkQeTXEn+DOpAsnoXuLEXbRtmfVOFPruHNCt+/yZxdRhxGrYHLJqkLgiDNIUrkQrOqcSB3ArsVa1mvhHlW5ZVkGjm2HUcDhc4oBjmWFKaAQq+FIFg7vjfPZiQu0PRu05Rr6VRsZxKZdCTFkVXWihwqYUIVhtlAbRdbcW+6J3gjhVXld0/f0bHaIe2QKVI14tk3EDMJSAXEX9N6jtnvHUum63lPJsZgO+0b5l5LMNEJE5wBbkVcFTK5SBVxMs1rbbaLNcbTx6krxVpPrDpPVJWDhlQ5uda5LXQ7CaWvJfVcJZQb3lNClSrStpZxdJft0XLCn9HO2kxj1CjeTytXoE0/Tu8wVW9S1Nqxi+fqVLVZrQ0i58NJLMEF8vUjXXngHLKUY871E4mCkJM22lCyfw+sNTYFxehbA3e7gbZe4Yl1DA3siVwavuEheIHqf3JaxCOgQEG9yENMM2IoxodTJLXawNfdNvFAiD0Qt9/VyNDXGWqBpJ5OjBaIQrAQEQt/7rsQtgC49C4CbEASo9MkEpRFeXJaoW2BrbwVw+Npnyy6kzj+tF1UKCqcTDd12xXUrBRuCypTYegmFhU19Jt5Lm/pyELpWFJLLoLw7wVhTB5IpiPzJiVY0P0qsL87QojZThl88r1LDDejyiRUnWZbDGGZYmCLo3PAM86usTo7Tq4D20gSnCtwPm68ZNwvna3Yloy3SQuFR6N8832MTqs+w/RvcY5FnSsjF36Avm2CCIp5IklQbsZ3mnFRzZsKjEs4Miucc3FsCjLKM/5iMitAwezFFfOJ5cJadaZUoNT4gvG3G/xKTTvefl+edlWf0wkSMkG3njnB2gPzaoLwiLBvFqXygLlKA8us3ucE/gsVUHPR2ZBn0yftG/CeWf4jUbLNLI/MpvJjRMs/Ivdr1PMVEGZbAMFMYCscrp7a7+FRFwndPO2+Pu8HFp+DT0QVQfb/7AVJAFmcrJc5Nq17EnMLdpoxjLwvrqYa1tK6/QN2WouKtkr0sw9NZNu6Mo5c3xJzhaHFny0aPJN6Gd8avpS6SiEZaeIBeRAkBVbuZTebTaNAAGnwlOsTecZSkNMtE2sjaEaLBqEDGgvPuf30kNkY8SiQZtVVzN0qdS2eMEHedKQf0Ael5swl2PoJ9RaJSYIMC4k6enR9dfglOPh5fXpHGPc1RhOCf6N80JSP5oWLFR6zJBSMT6YZUWL+J0pp4AbNBvCct9pSQNrgmLhGe8zcxYsLz6T8t+OfJvPoGWBGIFCvyQ8WKc8niH+MrZ7vQ5Io9e2lehaRPXvJa7AFMsxrgzSvhw5hmlftNntEQ+zJ3rsQlWlLzpfdVP92nfVtua96Qcjg3szqVxEIHwLJHST6egkfGxaeAYPzIiIePphDclXdT1B163lUsP48UEhcLJLxSfIUUNT3L5sk0XRgWdtFwb5eRXHcpWdf1Z+nlDe8FF632nowTTG0+R5ZT7jCRtd/Gzcqv4Ke42KAMSLvL2bm5mUU38BzjNJqtYU+c9LglaehMcznuY9OqqYL+8uB8C7jW8uBaNmFk1GdUbfYKJmb5HgeOtqE2dQn0l2KRKKlNt8vEyl8EK7+hNl0cK98Nq9YiWLUaatPFsWrlYQWPSkGYCKYuJ5OGEG+UdUbnPSaLe0yI4Hb8OLJIa4DrueVFFttX85a9UhfEbuBfDlVfMr0sCqqWqu8xyMQgYTsLzZAIKw61VT7UljLUlkNV16Eq1X3X6owyLU4Z9q9voRCIx3K0gZaFVGlytOUxiAHRjs0tCJ4ccRyO0jhK2sJWbzaEnU3uyFTaEHZyFM5uolmDjwX/TFKXSCt05n7PAvhwzGiRMbhRIF9fBVhLAGutDqyJwDKlzo8OE4HZYvp8NExPlfK22lYccbMaYzqpx31ZS53MLLQl0uXUbMk1W3k1UcbaOFxLKdCo2kaPXC19UrdgwhgX7VMMHI6UZ05yNkXonbYlHky6nqS3GI2Xs18JQfGiTn74r9S/LRKGAyhu+sa2y8ibWl21F94xmV3CmTeI+iGLpk4fPPpODLzMh/c08OUq6sdmN9CuH9i9ZgkYb4+5OWtEfzG6lCyDN4kI0zNp9akoPZ5W5BBAlKs9lwC7DK7MAprDE3D43PDEnUVeM3fD2OlWldyjsYtVNuWrjZ0u4UgSWHbbxmIjou/TmYUyMms+U05lhBJsgMa5xMoGRfvYLeS3ugEJ8qauxzKO5eDC2pSjI/torBFZRQOODW+zZ8FGEjuMBXfHidTPl6WES6VhTYpQCHFyKLp6q1V14I8BnZgKi1Epb8TG5NxkT+qH+ACcJGtWL8AGZ0pj5hkb6671WVi7jo9b+z3RH0GUB8iDT2WAswwiZ3pXeigI8c6yqCi3+4n0qBD0wyN8gY7NQ4u9MkdfVctv6hc3xdnNLq2iuGTyk5B/KaNju0gtYC4CyDYxCkArTIziOukzTIzqXTF9YhQ5dBaeGPV7bvkTY+FltwUmxuyS4hITI2/8k26nPs98mXFn5fkyE4Ml58sMlx83X0rS6DRfKvWL5sslknHkIFY6i+pVHbhmQIck9KY9Km/ERurcZE/qh86iSKiyWVRlU/ksatRnc5+Oj1v73FnUAOc+iyo9FMyiWcoxfRZVIahPupGv4yhJ2vBKQXwHKYggmIPeikMTwq9D8GQiXn8UJtnachReR6OGh2dgYHauaihq1QadNhse/r1N/97mf+/Qv3fq0rYj9EvD5C3XZ9DcYyeqOc0aFdj7rFLwiPg+6aF/+u33DKjxToQlg2iW/O595/ygbkwi2CceNuPQ8bi5MDctHKnP7mhGqMuTvyY0wIZiRKbaGRzWszcYPZ8Y73QWevCATghz8elhTnKQlCwBeRqzxamFbQtvg6vwd2XerOn5bTWGc4GVQbjMpBYQ2SfdXQy/lw2dV+EDlz23eFzamlWxtbZNO7y3hjwMW5t4nLVhfeS0KctsJogboOziNpvAA80UBV/PyXOWpcHLbAyzANloNCD5lgZdSWpkwrsJC0elN0Qwo55hDQpyARfyRcDS9V6146a2L5pKWF0m7Giiqfk5QDGslot7VVRhOGN+5ZImvEpVewNQFz2s1FAQ1VtogpfbwknsaDNN7uhg1rKBMRKaUkfHlImdMoSscb604WNourTxG0n4ClmRuJk3jKQmHKmiDNl1wXNoWMxyzBqZcbykAatRxm+o05CxLuF2Tn0nXmMrjdU4ijUxoFxG41hUPlNUtJYV6RAF+DsiPI1mdKqkYceJVwPwyySe1i91YdpWu2eNRQ6etJmwYWZkamjpvgRNP24EvpEen7ebN5ZtB+zGlsFBurRojdfsfo/upqnXol4kfes48YbzEWZFSeIBBqjKuQbknAFeDa6Q0C+WlG5jKQx/q4gjtKZLXg12n9oenSoEJGl4eGnLmpXCfsWDpmPbJW3p5XwgM/n9Uk6HWYBMQWoMzFUhKpa8Ui1bApV+DRmK2Yvd6dKi8jSG2J89pywpWqDKNZyjVCWgb6jD2VqkgYp7XrKSnPZ7rEPIG6rAMSGU2Uz0yFrBbD5Gcyl3Y5FW2WLSdNJ6JkwFgIsfvQyCy2OmL34t9nfbYn+38wVn+5ns77bN/m7bDOP283bzxpJ5B7uxCaUTt7YVbm0XbBzQ7OM2Tm1rUy1kQyPLS76zm501t71umDx4e94JJisnP96T5s+Q1X88VLfGIBWFSnjsMIDrsXbxEBnUsU6unCDNCqGIGgYMN0loqnnyT7oHRx9P6tYIeQnh+krAbWHzEvDcq4yu/QgdadqEd4UBUMD2qH8GOBd5wfEygSi3GRkooqDjoVM1B+Zl9fcALBzcSGJs3+JkElgsnk4jQjBl4+GVSvnI6/KRZIpkH0eZ8dKePcCLQPyv+lKQ6H4FhWNuLKknkVrnmk3MEHGAoW2TCAjm1sipkRQgs6xeTeSGB58PQnIsccl0cSq9i0iB4abv4uG5OXGI9FmJwsxX9hxJLOTbfIeFFeyVJ8JdDrD1cpBtRpXJZUnVRUdOKxXRhoHJ22SS/XC5QUPp3303T3/no8dPG2TgeVt5RmNZUjUATE73F5JSggd96guuvRVt3/0IWe0vJan7dnHaX0lKC4E+l4T2S+Szv5h09leWzb5dMvulctnPl0qjsfpYyh1kgsI9ZZYVKssuTpxWHoO4lUjCirXxsRpRkedgOXgYh3dxnzu7bXFuDUOxX4KgvaqAcu4q0KNNbCCeydHfRqJu6QBKeE11i8I18pVlxhlaEpqdHtqzmYlM8rhg9DY1X1lCKm9RpuC9ygULrbcl71iQ0dvdVAl43QWK3SeVoNi018LG7LzKwOAZbgjY5AwFIPfKEq/jeicgA+p4uK80KKVYbuO9XekuQAH78jVMOVgXgC2Xy/TmJWdMRm+Fh00mcKqd+Xjrax12X8rPlLPkGo96i0d6qIv0IC7xWBHW7+7wCxIanllTp7s7Kgby1Z2y+an0xk5Zvp3VlYrjXaBTYmiOKiVAOmqUXN9iYeA1Aafmkkq9zAGUd+9GGWbpxSij+qKUWeRSj95GvtjDy9wBFN/sEfAWvNYjd1F2tUd7U8/2+pEKz+YjXc+H8I94E9KL72BFG6bR6IE5MBvUHxIpHKT8JyJ1lvqwJGRxGM0HcCMF4mo3zk7fHmzAI3J6ehB8zk4sBzC10tuPh4fd8+D87BIj4S8gvLvaPe7uwyvT2avV9ClS9ppo9hC8FPVdsM+oRbsrke7lhgIIoRoKGjys1+Nh4/Yos7KsPQQAYWZBc1JavPHo5ExleO7ysbl4U/DMT36uUcSPV3E8u8kg0uSiYnW0JuWBqLuM5E3OSPKr876d6u8J8MQ4Cqzt5pCzABeqJWxAbi/EBNx7LmYCr+LIhAyixoSX7kyAcTizgFXm/TrUVsgv8M0lP6r6rmc9lpbenQo180aPQRtsR0M1bzzbIvA+nswT84mk7Kg37+UtNptGcEiSzz2pViEDHTLMaJ3m1nmjPSSB2jfDZCF0Mx7TiPA8eLqxtqNWhrBMDCo5OYkvnnGcez9gnPB/LwX6uVVIVwuQyeltUvn5XdxaEdZqTaZG3d6F+XKpAk/zK6oUIJl0eScNSx8LqhuBX6xseICQr2hv8kws/B91gkp0Lav0HKqWQXPWtJc/QdMkSlBFI3T/kWPc+wFjFFpGcP/VOgYT0ppEiOdRMIRH9Qv1Ihe+3S2Q/Oup9Gq0NmttlGhU2dQFwyifunCw7vokw5ZyqjMfkD5DvSAM19lEbraX07Vk3e05xZ/TYEO9H22wpT6WtqslgpBVWkAOJMiyGLxcQAxUEG6mTmqzl9OtMD0/QABka/J83LdbE53z/GWRoxPvehLOBgkYiql8osfsx5b6LiSrDN/4+5I4f9OGs2iAT5I0oCm/hcL+gBsd9KUpUnnyPb6L0wey6L6bQjbRdfGoW3wX9OczvCR4RdhA/uc3vFYPOhoSUb4l5m1tz+uPJgn5lU4YQnJzdpXiXsmxwJ9viiEzBbQQR/xyU7y7UNaQnutXCp+xyfKMFOWglKvBqobAOToJ3p4RSl2IdPr2hwFZGkYpJQfiBrjngbiSuuspb2yJBHMCypqHkev6c+dbeC60SSPptarEdKgJNsmABPAy9Flaw3s1RZ9orm3KgNR4u4q4XK3BW/ZZA0CpVzH2fAuuBXFOZxJUvjcpgcyJE5NvY9UNfPIv9EjYOF7sEeCKMGHxabYbdHTzVteh35HY5k05pTZTG0vdvL3PrMvfM7443e4Rvf/OCVh0FcfU3jSOZobRBLwbGSINATp79Wo+upuMw9mD9xbhCF+LX3HQt0Q180lj59mj98RKppN7YBZUERMVXiFmaiVSPtINygbNC5yphpTIGHUSUmxwtcdExWsevBRFhCMNa4PwIdn1X+NtgwGkzNcq9/TubRZYr8NOn1rbm3oJTR+w67GSvFsfZLKIU9uZd74xkKmjZxP9aj96aLKA7J1AAlC47JPe2flarE3Acv2ahjQqkQZDI+xVDB7A5Kukh7SZfQT8vT0ZRN02KtZ37iEe74Rl8pYwlU1C7iGOxnsbBvxEp+z4jtRreFcmTEg3osvR74w4vdK7J8oZim4HFDzBG9KulmAfDY+em0gu0g7NQ/6a6rGyxNqA/VEiFHadRhjQ9jVka4uTtJZdYoK5CrQhOO5cXL4O9j+ef+oSln6u159TY7DzhdUEWwWTr876IR2MCJ/ZOCwp1xKdHktoiYWkyqB+sIJI3b/tXHS1zhfWDQ0cvU7Hvxx09ztfnk83KIq4uDBXFapuSEuIo6EHx3K4w5d6PsY4nR8dsr+3CAHCb0QywySJb/C5Wayd3b7CujBPktp9ov6FN6ZmIcze7LF4a6AlHhFKeZBn8bCkwdEhr692cxeP7W4ZL9U5CV3lN6KFcpuKLZlkNj7LTUPec7246V8KmlovzEtQMnJZglHYIOqFDf+S39DaOcEsIEITFJOPoO9Dla18zdUBYZLSQoIZ1QvGp9Xey4BDvk/21TySKjlahob3ZGVcLDi8hoM5VgBSCsh4uzaT7fnRYbXu1G7P6E6dBRCSuYdSunfC++AEp1lGpZ4LvH9LY93vlwHZrkuIJ4d32GIdnojx+qN4HPfRIxg9eLVoFN/E1/AiK95b2AA7tEEGjFcg2NYGbYKrj4A6DDiFC+Sze7O04nw8iGbDeDTCJs1qr+FY1Xev2nKvuuVeddu96g6v2itLVcqaFyR6OYTpVq6Wn3pNdUWkJkqB6fln5xjKPRP1ZGP/+Oj0aD/oHh+9O4L3RCynG7JHQ3u/CaeWUa2RMTFwZKH3rnsZfOie44JNGitpmj9SGfxuDiy4lC3RQEJOMMtO8VLUMmbnISj7PaI29X4y1HsSSqg/lkBtYAGUZXxp5iGIj8Abl3wQcFF4sjGmmyh4hFYlRjkjIjGRUF+fH9XOy8G8oWBsDxSooBaIHLVYpyveSY/utOTimLFEelFVqZ1zz18jqJq1ShbJhifJRtZdtk5Tn3KhRDthaT8wzlAb9qKPTV5MRs7PxMyidD4bUyQaMAOiafrckAdfqQyioXc/i9OIfUlqCfQxYy2kurCyGQ7j723IadHwJvN0Ok+DQTzDD0RWq+tVJs4TWgDvtoTpbS2rWpeL1+++kv/WpuEsGqfJ7uVsTpby0XeyQCWLPvyTVv8cjCZ99J1htSrvNzFKvJ3Ho4FHlgzRYD6KvDTEKXAGc+QuXS1D7Ag+2MLmTprwgzdQ0qzIdp92LOkogYnXAj8ToBf4Hktbe8lrPSYkg5RZROKf1MRhpY/lqLaALlbw5vLpZBwZD8oIQy8Me9uanAHZuf4pHM0jYw6hr9LlHLJl/WNl+5ksZDdRncnJ/RVu7JN2HESmd4zkWZqae6Y7A7i+PB2sH4RpeDgj7Wu8Lt+SjMdpbTBcv43CQS3TuIv53R1shtLEMJSv7JOaPWcB6ovsO4/VQ56wHdPxVxahP4VyNePEAPUvYUaO/EnrZTo0YS2lbDisKLAQkpbIdKz+7/E5v+dC16GwGObvQ9XbVblyBjmLv6N+Jy2JWYJ3ZimZM+quVJpLZeQeLSaaoRdD5UEiwd56vgLlekoV8/aGMkh7pn9H66Dm0K9Sku4DEUgzZdZRK17CewtQJ8ftUWt/FLNam2e/dfO+Mjgsvb8ycouYKeVMUvh0TkVsaYXkeyerioZFDJ9VKug4hVSo1qPBeQzikZCPWFslLgWQT1xazogrVJlpIzNb1w/0UFzRYZ15dSuEUlOgo5mZ4ZC4q3SeVyfXYBrSx8mYS7BBvHJRdv2AGAWP1LN4Wu8n36oVwfHE1ljbP7O0VcdqAaFWoGTLh5OPilrBBEBmq3QSkD9rCjUaRLoG0ffdQ9Iq0o03byH1aqmvs5O3Moee31gwUWtc3DOVg2wiJMLzmbCE+FuPyiCfGt6jBIn8WZUaPZqIQgMTgyfaKpMz9flEj7mxmU9Pnb6rXkV/JXHKn7gRFoh4nbMHfL/W/o6LfouCwMUmV/wZF2mDlz7TktXgL7gYDxmycnxLRbmloJgyDiV746Vnc9XEkAuenMmIRUyOGB17dqdR+kSNpQYwDWDZWg+iNIxhrhlW2WOLnnjCyMvtk670cLa0vgxkPjmDzVhA9CLtnuouNG/9OJq3fgHNW4vRvLUkzVtL0xzfHXpuaiPQn0vsvC7zad1cktbNfFrrzzAZdo+m4ibe0rVh/KT3L9raMaPual0rXMXARP3i7uLM014gKuKecpctn4mu/rfK3CrmeseXl0LxPDuP9a27sVh/TamQwXDWalOaEn5JmdZ/Cb+0hzF+Mb9oDo1l+aU/8rEwv7SEvxbNkxJU4+4anKaRHwZbpeTWbSVr+De6DJJZy4Cp+cFt9Whf0kkxMQ67AuoabydLCVR5BinRc30XiQkl0fI2GbMqP7KxPK09svE9ucmAkZXczSrbkpkT2uVqtJsAsDzT/zESIKfdLhIB9izDL5UBJUX4AkKgpxa3ScGCYkCzUP/HSIGUkLtMCLaDXy0GcvbwBaRASzqeKwQvMDkpyxFumS/obpZ15mdJbduVgq3+6/pzcE3KuPvr5neWwVtKqe7GRTldcCEDTdrlWu8SrmyXc+U51jlSat1fx5btVdiy7cAWhxbSkkfJCuyoSHqizZ/BO63PX8c/Qqy1PRYV4riq0cjlxr3iVm66pWYz/Zlswh5/NZMgYGYxFlFCLcYgW5tMwTCcN/oe9tO1Fk9HKC9TrXzDu7zYqPVz2CZ3WOhiVJ+dVUig/oSMKiS+B+IwevAc998UQrmxraBJxjWa1Hc8Ga9JKVNljbOyzcjK6sC6lX0NvdNnULpiN7HKiZPK+WTd+GVQyI1nJc2elCS3i3Kt//N51v/JHNtfll/9ZbjVd+FVycxluf77U+yg0e2vm8MgwAJwwcCLXRGgsbvlaBpNCjoayLKG2uRGA9qtKecyJpPJEgMzsRDep6BxXPaZTwmRl9jOmtuW6LysLiXfTXhMjFGblyn7uyp4yKkFFFfBPIPuGyP8ZRI2rD4Stjzp6X3u4iTBeB+IB3OfhRWmLTAVF7TLBM2S6tmr0S3rLOuBk0tszcb4M0yLreNfZ1wsBF1gy1+nnxu7HZo+aeGb8LwYy1ohZaqgSet56KbCdinczVwNWQNJlUMgm7FQou/oORBWe46zO0to87O4AxLd6Atuxf6BFv6uB8VSSIXy0dreLDv8AaIVcdvIZwK3xQkSGc81/c4L/Lcx3p5JQNohpkkN7DOLnuEAN1bsVdWcBkRaZMCGFaHkc8x1YkvRsJTI5XTwC+chSVxZSo4aDHX3Ef77VHwYSX8UCp+tinrDIuPocjwSr9T8GA792t0UhT+oYD+DO7lh/Fn4/jNMB0bs9C+fDnSMnmM6kMlXcgREKbb7qIyZfqw//Q2TZ+8+atHt5DveG1AKMMidlGDKC6UEw/uLj5T0aPwVppbTyXjtIh6HMZtk0POJwtnowdiOy9xGnhjC3OKZjIMEoOGNBzy7DBDYTzlXyu/9F261avQl64iMxD/SNvz/qFzHqFyngDxzW8wI8WubwaU2AykKFTOpNnljLKRFWjHJ/OI7lZYQBlvEIyYcUfCAYdV5dhX4Qw2ZkGIRLdEJJbqp7+fptNIiSnL3zxfVPcVSS0zV7bUclfLkpRj+svtoj2J5yniz+6iw6cnElqlu9WV1/V+TeFwjRKjdpywMC82lhTn6IBTbDb+UwJnsZrolLouzRXp/3BYtAd8zFjGZtoQbsG5EzAGHl3uLf1gVV/d5XQgfrhTHlr7wLtLwhqxlkEttseOS0MU3S7IIb2x7k2kKoa9YXAKVuEDBP7pf6KsL2S3Xhidd6Zb+aMEfxpbLUwXT+Z6dHn/JgOl3Ss1bi/b9jCcWHEvZHhChgBymN4Rv8N82IWrKQpToE9g3keZwswDcx69trwYTf8Ct3dceNMFXeMWw6eNW0vu8cosnoxt/uW7W52PCippKpbpL56wXuaBSIUr+rtsMLo9OupCAbXOTfvHFlx32pcW/7JA6px9Pgs9n5//ongODXlfOPl5++HiJvlFwcHQOkcQbYEGjcboxmMXfoo2ThwP8d38yCq+DUyKz15PJ12rl/ONpcHh2fECU/7SD8KtVjBKJUi9av1n3qmGawgvwwSu/Svy9EZEcL5ynk7WbaIyWEzKCQ3o6Qti7KZHp4WREbMZ6hfJ+FqGMB1kAM75gJIKeybJ9Pg5oG8xmKUKhIcXIHvxB+cTyb2q110l5PK3VK1KiSoSSPfdI2w2r0PIREIU2Q/hR++tvX367+20Q/Pb+t5PfLv5aZ6EyjFFgyzD4miNcJ8YEw+LY0AjA9HYWRbJQNwOAjKLdoLLmG19a8hctCJzpw81ock3mRS4maJvYkyQ4BES/zrBNaFxMNiVVJVTgYk72V0Or4yt1fGudllKnJdV5osaZYYmR4rqmsxwxDAUWSc/+kgPq6RfJ4yDQriEonbapKWDo5EZq9Ke0eH1/inH9M1u99WkIkwWxTMk6XGAD/IOYeM4RUZEBOIkSgYqbE2+BNAtnxGjeT2ZfI3zjUlJGpXUapnMJOoOGeNZkeshsvKoqras9sz0WnOLsJ1e1QhG5EiyARKIF6iMo4Yw4PXlNj3bm1bIJCqJ66m1i8HPwtYIRXVkaSiiyuZ4baUFClk63Jrh99oHYws5xI+P/YbdzASGmcqIB1g+oKWM1UgEnu5qkRd4afzFESmjEbAAFQvHScjzIxFRF2hTmBlfJqqzwu3JWB+qfx9+jAX+k6PFJXokGZRG+JRmqMtCYL+2+bkoEFYRsJHrWv1Jl9yXt8xVl9w1l90uVHQZEUYVF+AiGlg1infDiLqlJg5R7pY+Si47VgdA8hyPuWXJkXEyKv5hJ8TWT4q9kUnzNpPgWk+LnmhTfMCl+nknx802KbzMpvrtJ8S0m5SXbKQB5ME2LX2Ra/ALT4heYFv/Pb1p81bT4kmnxbabFLzEtRXrbkkS8pehty9Db1g/R25aut60yvX3hvYfoZSoHLQ9z0XOhIJSb4wk2ruC8m3k8CMf9qIHsDumedXY7oOCxuTKb6phDRh/ieyUA3FJTz8PQyK/sW6y3bwCwQJDtX8vR/rUWs38tzf61VrJ/Lc3+tSz2r5Vr/1qG/Wvl2b9Wvv1r2exfy93+tYT9w9U9RdYwXa0im9cqsHmtApsnyLe4vbOYp5ZqnlqSeWrZzFMr3zwpe2xs90ZPekVWXnm7OBR8tocjdtwZtHXMh6WYHGNHaRaFA3XHR81qohEZsysEfNsK5GA4moRpjcG8EjtavXXYrKQvcIPt4ftLsKB+YHsG69KTv8abQKUdi7xWalsmMacTbxiFCSS0zIziEO9nESlg8rQu57516q+y2iSoTYAAIGM1QDD3DLQNjoan7VvUK3w/7iT8TrcjiPGvt+kYm7uP0j7LU4N+9dlXX/naYl/pfgts5nHIZ4gO299oe48q3lBTLMX1rQGpd9JN1in7o8X+UCHWK/GQr+7h6plCaZyUsKytbEdeYp4fUhNoQEnwyDj1V6n9X3uA7/8BJvIdogxqAQA="""

def _get_scheduler_source() -> str:
    if SCHEDULER_FILE.exists():
        return SCHEDULER_FILE.read_text()
    payload = gzip.decompress(base64.b64decode(_EMBEDDED_SCHEDULER_GZIP_B64.encode("ascii")))
    return payload.decode()

# In-file defaults (used when CLI flags are omitted).
DEFAULT_MODE = "bound"
DEFAULT_TIME_LIMIT = 600
DEFAULT_WORKERS = 8
DEFAULT_OUTPUT_DIR = "max_penalty_outputs"


def _load_scheduler_namespace(maximize: bool) -> dict:
    src = _get_scheduler_source()
    src = src.replace(
        "import pandas as pd",
        """
try:
    import pandas as pd
except ModuleNotFoundError:
    import csv

    class _MiniDataFrame:
        def __init__(self, rows):
            self._rows = list(rows)

        def to_csv(self, path, index=False):
            if not self._rows:
                with open(path, "w", newline="") as f:
                    f.write("")
                return
            fieldnames = list(self._rows[0].keys())
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(self._rows)

        def head(self, n=5):
            return self._rows[:n]

        def __str__(self):
            return str(self._rows)

    class _MiniPandas:
        @staticmethod
        def DataFrame(rows):
            return _MiniDataFrame(rows)

    pd = _MiniPandas()
"""
    )
    sentinel = "run_output_dir = resolve_output_dir"
    if sentinel in src:
        src = src.split(sentinel)[0]
    if maximize:
        src = src.replace("model.Minimize(sum(objective_terms))", "model.Maximize(sum(objective_terms))")
    ns: dict = {}
    exec(compile(src, str(SCHEDULER_FILE), "exec"), ns, ns)
    return ns


def _ensure_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def _constraint_bucket(var_name: str) -> str:
    prefixes = [
        ("clinic_underfill_ge_", "clinic_underfill"),
        ("rsch_exact3_", "rsch_exact3"),
        ("pgy4_pulm_pair_pen_", "pulm_pair_pgy4"),
        ("pgy4_micu_pair_pen_", "micu_pair_pgy4"),
        ("hard3_", "hard_3_run"),
        ("hard5_", "hard_5_run"),
        ("post_nf_medium_", "post_nf_medium"),
        ("post_nf_hard_", "post_nf_hard"),
        ("rif_when_min_", "rif_when_rads_min"),
        ("non_sinai", "non_sinai_pgy4_micu_early"),
        ("pgy4_preim_medium", "pgy4_pre_im_boards_medium"),
        ("pgy4_preim_hard", "pgy4_pre_im_boards_hard"),
        ("pgy6_pre_boards_pen_", "pgy6_pre_boards_rsch"),
        ("pgy6_last8_pen_", "pgy6_last8_rsch_vaca"),
        ("micu_after_nf_", "micu_after_first_nf"),
        ("vac_", "vacation"),
        ("fair_", "fairness"),
    ]
    for pref, bucket in prefixes:
        if var_name.startswith(pref) or pref in var_name:
            return bucket
    if var_name.startswith("x_f"):
        return "assignment_linear_terms"
    return "other"


def run_exact(time_limit: int, workers: int, out_dir: Path) -> Tuple[str, float, Path]:
    ns = _load_scheduler_namespace(maximize=True)
    ns["WEIGHTS"] = ns["BASE_WEIGHTS"].copy()
    model, var, _, _, soft_report = ns["build_model"]()

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_search_workers = workers
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    objective = solver.ObjectiveValue()

    suffix = "maxpen_exact"
    ns["write_reports"](solver, var, soft_report, suffix, output_dir=str(out_dir))
    soft_path = out_dir / f"soft_constraint_report_{suffix}.csv"

    # Aggregate per-constraint breakdown from produced report.
    breakdown_rows = []
    total_penalty = 0.0
    if soft_path.exists() and soft_path.stat().st_size > 0:
        by_constraint = defaultdict(float)
        with soft_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    by_constraint[row.get("constraint", "")] += float(row.get("penalty", 0) or 0)
                except ValueError:
                    continue
        for c, val in by_constraint.items():
            total_penalty += val
            breakdown_rows.append((c, val))

    breakdown_csv = out_dir / "max_penalty_exact_breakdown.csv"
    with breakdown_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constraint", "penalty_sum"])
        for c, p in sorted(breakdown_rows):
            writer.writerow([c, p])
        writer.writerow(["TOTAL_FROM_SOFT_REPORT", total_penalty])
        writer.writerow(["OBJECTIVE_VALUE", objective])
        writer.writerow(["STATUS", status_name])

    print(f"Mode: exact")
    print(f"Status: {status_name}")
    print(f"Best objective (maximize): {objective}")
    print(f"Per-constraint CSV: {breakdown_csv}")
    return status_name, objective, breakdown_csv


def run_bound(out_dir: Path) -> Path:
    ns = _load_scheduler_namespace(maximize=False)
    ns["WEIGHTS"] = ns["BASE_WEIGHTS"].copy()
    model, _, _, _, _ = ns["build_model"]()
    proto = model.Proto()

    contrib = defaultdict(float)
    total = 0.0

    for var_idx, coeff in zip(proto.objective.vars, proto.objective.coeffs):
        v = proto.variables[var_idx]
        dom = list(v.domain)
        # CP-SAT domains are [l1,u1,l2,u2,...]
        min_val = min(dom[::2])
        max_val = max(dom[1::2])
        chosen = max_val if coeff >= 0 else min_val
        term_val = float(coeff) * float(chosen)
        bucket = _constraint_bucket(v.name)
        contrib[bucket] += term_val
        total += term_val

    breakdown_csv = out_dir / "max_penalty_bound_breakdown.csv"
    with breakdown_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["constraint_bucket", "upper_bound_contribution"])
        for k in sorted(contrib):
            writer.writerow([k, contrib[k]])
        writer.writerow(["TOTAL_BOUND", total])

    print("Mode: bound")
    print("Method: objective coefficient × variable-domain extrema (fast upper bound)")
    print(f"Upper bound total: {total}")
    print(f"Per-constraint bucket CSV: {breakdown_csv}")
    return breakdown_csv


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Maximum penalty calculator")
    parser.add_argument("--mode", choices=["exact", "bound"], default=DEFAULT_MODE)
    parser.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT, help="Exact mode: solve time limit in seconds")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Exact mode: CP-SAT workers")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    out_dir = Path(args.output_dir)
    _ensure_output_dir(out_dir)

    if args.mode == "exact":
        run_exact(args.time_limit, args.workers, out_dir)
    else:
        run_bound(out_dir)


if __name__ == "__main__":
    main()
