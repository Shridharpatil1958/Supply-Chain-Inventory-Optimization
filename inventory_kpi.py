def inventory_turnover(cost_of_goods_sold,
                       average_inventory):

    return round(
        cost_of_goods_sold /
        average_inventory,
        2
    )


def stockout_rate(stockout_orders,
                  total_orders):

    return round(
        (stockout_orders /
         total_orders) * 100,
        2
    )


def fill_rate(filled_orders,
              total_orders):

    return round(
        (filled_orders /
         total_orders) * 100,
        2
    )
