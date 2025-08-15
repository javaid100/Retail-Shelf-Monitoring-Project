def classify_stock_status(product_count, total_count_all_products, threshold):
    if total_count_all_products == 0 or product_count == 0:
        return "Out of Stock"
    elif product_count >= threshold:
        return "OK"
    else:
        return "Below Threshold"
