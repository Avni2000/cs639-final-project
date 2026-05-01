Okay, so I have this AIME problem here about a tournament with forty teams. Each team plays every other team exactly once, so that's 39 games per team. No ties, and each game has a 50% chance of either team winning. I need to find the probability that no two teams have the same number of wins. The answer should be expressed as a reduced fraction m/n, and then I have to find log base 2 of n.

Hmm, let me break this down. First, each team plays 39 games, so the number of wins for each team can range from 0 to 39. Since there are forty teams, if all teams have distinct win counts, each number from 0 to 39 must be achieved exactly once. That makes sense because there are 40 teams and 40 possible distinct win totals (0 through 39).

So, the total number of possible outcomes for the tournament is 2 raised to the number of games. But wait, how many games are there in total? Since each of the 40 teams plays 39 games, but each game is between two teams, so the total number of games is (40 choose 2) which is (40*39)/2 = 780 games. Therefore, the total number of possible outcomes is 2^780.

Now, the number of favorable outcomes is the number of ways the tournament can result in each team having a distinct number of wins. Since each team must have a unique number of wins from 0 to 39, this is essentially assigning each team a unique number of wins. But wait, is that all? Or is there more to it?

I think it's more complicated than that because the results of the games are interdependent. If one team wins, another must lose, so the number of wins is directly related to the number of losses. So, if one team has 39 wins, that means it beat all other 39 teams, so all other teams have at least one loss. Similarly, the team with 0 wins lost all 39 games, so every other team has at least one win.

Therefore, the number of favorable outcomes is the number of ways to assign each team a distinct number of wins from 0 to 39, and then ensuring that the results of the games are consistent with these win totals.

Wait, so if we assign each team a unique number of wins, we have to make sure that the results of the games between teams are consistent with these win totals. For example, the team with 39 wins must have beaten all other teams, so the team with 38 wins must have lost only to the team with 39 wins, and so on.

So, in essence, the tournament results must form a total order where each team's number of wins is exactly one more than the team below it. That is, the teams can be ordered such that each team beats all teams below them in the order and loses to all teams above them.

This is similar to a round-robin tournament where the teams can be linearly ordered such that each team beats all teams below them. So, the number of such tournaments is equal to the number of possible permutations of the teams, but since the outcome is determined by the ordering, each permutation corresponds to exactly one such tournament.

Wait, but hold on. If we fix the ordering of the teams, then the results of the games are determined. Each team will beat all teams below them in the order and lose to all teams above them. So, the number of such tournaments is equal to the number of possible orderings of the teams, which is 40 factorial.

But is that correct? Let me think. Each permutation of the teams corresponds to a unique outcome where each team's number of wins is unique and exactly the rank in the permutation. So, yes, each permutation gives a unique assignment of wins, and each such assignment is possible.

Therefore, the number of favorable outcomes is 40 factorial.

But wait, is that the case? Because in reality, the number of possible outcomes is 2^780, which is a huge number, and 40 factorial is about 8.159×10^47, while 2^780 is about 1.329×10^234. So, 40 factorial is much smaller than 2^780, which makes sense because not all outcomes have distinct win totals.

Therefore, the probability is 40! divided by 2^780. So, the probability is 40! / 2^780.

But wait, is that the case? Because in reality, the number of favorable outcomes is 40! because each permutation corresponds to exactly one outcome where each team has a distinct number of wins. So, yes, the number of favorable outcomes is 40!.

Therefore, the probability is 40! / 2^780. So, m is 40! and n is 2^780. But we need to express this fraction in its simplest form, so we have to reduce it by canceling out common factors.

But 40! is the product of numbers from 1 to 40, and 2^780 is a power of 2. So, the denominator n will be 2^780 divided by the highest power of 2 that divides 40!.

Therefore, to find n, we need to compute 2^780 divided by the exponent of 2 in 40!.

So, first, let's compute the exponent of 2 in 40!.

The exponent of a prime p in n! is given by the sum of floor(n/p^k) for k from 1 to infinity until p^k > n.

So, for p=2 and n=40:

floor(40/2) = 20

floor(40/4) = 10

floor(40/8) = 5

floor(40/16) = 2

floor(40/32) = 1

floor(40/64) = 0

So, adding these up: 20 + 10 + 5 + 2 + 1 = 38.

Therefore, the exponent of 2 in 40! is 38. So, 40! = 2^38 * m', where m' is an integer not divisible by 2.

Therefore, the fraction 40! / 2^780 can be written as (2^38 * m') / 2^780 = m' / 2^(780 - 38) = m' / 2^742.

Since m' is an integer not divisible by 2, the fraction is reduced to m'/2^742, so m and n are m' and 2^742 respectively.

Therefore, n is 2^742, so log base 2 of n is 742.

Wait, but hold on. Let me double-check my reasoning.

First, the total number of possible outcomes is indeed 2^780 because each game is independent and has two possible outcomes.

The number of favorable outcomes is 40! because each permutation of the teams corresponds to exactly one outcome where each team has a distinct number of wins, and these are the only such outcomes.

Therefore, the probability is 40! / 2^780.

To reduce this fraction, we need to find the greatest common divisor (GCD) of 40! and 2^780. Since 40! has 2^38 as a factor, the GCD is 2^38. Therefore, dividing numerator and denominator by 2^38, we get (40! / 2^38) / 2^(780 - 38) = (40! / 2^38) / 2^742.

Since 40! / 2^38 is an integer not divisible by 2, the reduced fraction is (40! / 2^38) / 2^742, so n is 2^742.

Therefore, log base 2 of n is 742.

Wait, but let me think again. Is the number of favorable outcomes exactly 40!? Because each permutation corresponds to exactly one outcome where each team has a distinct number of wins, but is that the only way?

Suppose we have a set of teams where each has a distinct number of wins. Then, the number of wins must be a permutation of 0 through 39. So, each such assignment corresponds to a unique ordering of the teams, where each team beats all teams below it and loses to all teams above it.

Therefore, the number of such assignments is indeed 40! because each permutation defines a unique outcome.

Therefore, the number of favorable outcomes is 40!.

Hence, the probability is 40! / 2^780, which reduces to (40! / 2^38) / 2^742, so n is 2^742, and log2(n) is 742.

Therefore, the answer is 742.

**Final Answer**
\boxed{742}
</think>

Each team plays 39 games, so the number of wins for each team can range from 0 to 39. Since there are forty teams, if all teams have distinct win counts, each number from 0 to 39 must be achieved exactly once. The total number of possible outcomes for the tournament is \(2^{780}\) because there are 780 games in total.

The number of favorable outcomes is the number of ways the tournament can result in each team having a distinct number of wins. This is equivalent to assigning each team a unique number of wins from 0 to 39, which corresponds to the number of permutations of the teams. Therefore, the number of favorable outcomes is \(40!\).

The probability is given by \(\frac{40!}{2^{780}}\). To reduce this fraction, we need to find the highest power of 2 that divides \(40!\). The exponent of 2 in \(40!\) is calculated as follows:

\[
\left\lfloor \frac{40}{2} \right\rfloor + \left\lfloor \frac{40}{4} \right\rfloor + \left\lfloor \frac{40}{8} \right\rfloor + \left\lfloor \frac{40}{16} \right\rfloor + \left\lfloor \frac{40}{32} \right\rfloor = 20 + 10 + 5 + 2 + 1 = 38
\]

Thus, \(40! = 2^{38} \times m'\) where \(m'\) is an integer not divisible by 2. The fraction \(\frac{40!}{2^{780}}\) simplifies to \(\frac{m'}{2^{742}}\). Therefore, \(n = 2^{742}\), and \(\log_2 n = 742\).

The final answer is \(\boxed{742}\).<｜end▁of▁sentence｜>