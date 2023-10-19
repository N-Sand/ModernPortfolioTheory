# ModernPortfolioTheory
A rudimentary implementation of modern portfolio theory, where we choose stock portfolios which lie on the efficient frontier. It is just a collection of useful functions. Heavily inspired by Quantpy.

Requires:
-numpy
-matplotlib
-scipy
-pandas
-yahoo finance

Assumes that a given stock history of returns is a multivariate gaussian distribution, and thus can find portfolios with optimum returns based on a chosen level of portfolio variance.
Note that real stocks follow a more log-nromal distribution, and variance is not a good measure of risk tolerance. People often use modified versions of MPT for real investing, although
it can be hard to find methods more effective and simplistic as this one.

MPT is based upon the following paper:
Markowitz, Harry. “Portfolio Selection.” The Journal of Finance, vol. 7, no. 1, 1952, pp. 77–91. JSTOR, https://doi.org/10.2307/2975974. Accessed 19 Oct. 2023.
