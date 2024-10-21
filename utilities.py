# input: [1, 1, -1], buy is 1, hold is 0, sell is 0
# prices: [2] + [1, 1], [1, 1] is given by the user, 2 is the initial price
# the user will input [x,x], with x being -1 for down, 0 for same, 1 for up
# so the worst possible price is 0, the best possible price is 4

# when you sell after buy, you always sell all asset, buy after sell similar when you buy after buy, assuming you are
# buying again the same amount of asset with the current price, sell after sell similar

# output is the utility

# as an example, if I buy buy sell given price goes up up
# time 1: price 2, buy, cost 2, gain 0, asset 1
# time 2: price 3, buy, cost 2 + 3, gain 0, asset 2
# time 3: price 4, sell, cost 2 + 3, asset 2,
# time 3: sell 2, gain 4*2, utility = gain - cost = 3
# assets will be [1, 2, 0]
# costs will be [2, 3, 0]
# gains will be [0, 0, 8]


# another example where we short sell given the price goes down down
# time 1: price 2, sell, gain 2, asset -1
# time 2: price 1, sell, gain 2 +1, asset -2
# time 3: price 0, buy, gain 2 +1, cost 0, asset 0
# time 3: buy 2, cost spend 0, utility = gain - cost = 3

# I will only get utility if I realized the gain, although I don't have to realize it at time 3.

import pandas as pd


def calculate_utility(actions: list[int], price_changes: list[int]) -> int:
    initial_price = 2
    assets = [0] * 3
    costs = [0] * 3
    gains = [0] * 3
    utility = [0] * 3
    prices = [initial_price]

    for price_change in price_changes:
        prices.append(prices[-1] + price_change)

    for i in range(3):
        previous_action = None
        if i > 0:
            # previous action is the first non-zero action before this time
            for j in range(i - 1, -1, -1):
                if actions[j] != 0:
                    previous_action = actions[j]
                    break
        if actions[i] == 0:
            continue

        # case 1: buy buy or sell sell or at the beginning
        elif previous_action == actions[i] or previous_action is None:
            if actions[i] == 1:
                # cost of buying is the current price
                costs[i] = prices[i]
                # asset is 1
                assets[i] = 1
            elif actions[i] == -1:
                # gain of selling is the current price
                gains[i] = prices[i]
                # asset is -1
                assets[i] = -1

            # no change to the utility since there is no realized profit

        # case 2: buy sell
        elif previous_action == 1 and actions[i] == -1:
            # calculate all the asset before this time
            total_asset_to_sell = sum(assets[:i])

            # calculate the gain from selling
            gains[i] = total_asset_to_sell * prices[i]

            # since it is a buy sell case, there will many non-zero costs but only one non-zero gain
            # calculate all the cost so far
            cost_so_far = sum(costs[:i])

            # calculate the utility
            utility[i] = gains[i] - cost_so_far

            # since we realized the gain, we set the assets, costs and gains all back to 0
            assets = [0] * 3
            costs = [0] * 3
            gains = [0] * 3

        # case 3: sell buy (short sell)
        elif previous_action == -1 and actions[i] == 1:
            # calculate all the asset before this time
            total_asset_to_buy = -sum(assets[:i])

            # calculate the cost from buying back the asset we owe
            costs[i] = total_asset_to_buy * prices[i]

            # since it is a sell buy case, there will many non-zero gains but only one non-zero cost
            # calculate all the gains so far
            gain_so_far = sum(gains[:i])

            # calculate the utility
            utility[i] = gain_so_far - costs[i]

            # since we realized the gain, we set the assets, costs and gains all back to 0
            assets = [0] * 3
            costs = [0] * 3
            gains = [0] * 3

    return sum(utility)


# test every possible price change with every possible action

# create table utility_table with columns: price_change1, price_change2, action1, action2, action3, expert1, expert2, utility
utility_table = pd.DataFrame(
    columns=['price_change1', 'price_change2', 'action1', 'action2', 'action3', 'expert1', 'expert2', 'utility'])

for price_change1 in [-1, 1]:
    for price_change2 in [-1, 1]:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                for k in [-1, 0, 1]:
                    for expert1 in [0, 1]:
                        for expert2 in [0, 1]:
                            price_changes = [price_change1, price_change2]
                            actions = [i, j, k]
                            # print(price_changes, actions, calculate_utility(actions, price_changes))
                            # insert into table
                            row_to_insert = pd.DataFrame(
                                [[price_change1, price_change2, i, j, k, expert1, expert2,
                                  calculate_utility(actions, price_changes) - expert1 * 0.3 - expert2 * 0.3]],
                                columns=['price_change1', 'price_change2', 'action1', 'action2', 'action3', 'expert1',
                                         'expert2', 'utility'])
                            utility_table = pd.concat([utility_table, row_to_insert], ignore_index=True)

# # test cases
# actions = [-1, 0, 1]
# price_changes = [-1, -1]
#
# print(calculate_utility(actions, price_changes))  # 0


# add one column non negative utility to the table, which is the utility + the minimum utility to raise the minimum to 0
utility_table['non negative utility'] = utility_table['utility'] - utility_table['utility'].min()





# add one column to the table, the normalized utility between 0 and 1
utility_table['normalized_utility'] = (utility_table['non negative utility'] - utility_table['non negative utility'].min()) / (
        utility_table['non negative utility'].max() - utility_table['non negative utility'].min())


# add one column "risk-adverse utility" to the table, which is x^2 of the utility
utility_table['risk-adverse utility'] = utility_table['normalized_utility'] ** 2

# print the table in markdown
print(utility_table.to_markdown())

# save the table to csv
utility_table.to_csv('utility_table.csv')

if __name__ == '__main__':
    pass
