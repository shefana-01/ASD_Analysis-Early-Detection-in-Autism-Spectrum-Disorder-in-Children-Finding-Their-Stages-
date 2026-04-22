from django.shortcuts import render, redirect, get_object_or_404
from .models import Product

# CREATE
def add_product(request):
    if request.method == "POST":
        name = request.POST.get('name')
        price = request.POST.get('price')
        description = request.POST.get('description')

        Product.objects.create(name=name, price=price, description=description)
        return redirect('product_list')

    return render(request, 'add_product.html')


# READ
def product_list(request):
    products = Product.objects.all()
    return render(request, 'product_list.html', {'products': products})


# UPDATE
def update_product(request, id):
    product = get_object_or_404(Product, id=id)

    if request.method == "POST":
        product.name = request.POST.get('name')
        product.price = request.POST.get('price')
        product.description = request.POST.get('description')
        product.save()
        return redirect('product_list')

    return render(request, 'update_product.html', {'product': product})


# DELETE
def delete_product(request, id):
    product = get_object_or_404(Product, id=id)
    product.delete()
    return redirect('product_list')
