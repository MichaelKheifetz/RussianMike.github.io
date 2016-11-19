
title: Classes!
---
This is a project I have completed on Classes. First I created the OnlineStoreManager Class
that holds all Stores and Customer classes. The OSM class consists of an __init__ method 
that is required for every class in order to define the methods used in that class. The
init method instantiates both a store counter as well as an empty list of stores. The 
create_online_store method includes the incremental store counter and an append method
that will add new stores to the list created in the __init__ method. A new_store variable 
is than created for storing the instantiation of the Store and it returns new_store when
the method is called later in the process. 

The next class is the Store class. In the init method for the store, an inventory empty 
dictionary is created as well as an empty list of customers. The store is also instantiated 
here. In the purchase_inventory_items method, the item and quantities of the items that 
will be sold are instantiated. In this store, the items are for a bookstore and music store
so the items are various types of books and music records. Afterwards, the inventory
purchase method is created by which new quantities of items are purchased; they will
be added to both the stores inventories. The next method is sell_inventory_items which has 
the same function as the purchase_inventory_items except that it tracks the sales to the 
customers rather than purchases from the wholesaler. Because items will be reduced due to 
purchases, a similar method has been created to purchase_inventory_method to show that
after every sale, the inventory count decreases by 1. In the create_customers method, we 
once again setup a customer counter in order to count the number of customers visiting the 
stores. We have a similar customer append method to one used in creating an online store.

After this, we create the Customers class where we include an init method as necessary for 
every new Class. Afterwards, we create a connection between the Customer class and the 
Store class in order to keep track of the number of items that were purchased by customers
and hence removed from the Store inventory. 

After completing the tasks we do several tests to check the functionality of our classes by 
asking them to print out the list of items originally in inventory, checking to see that we
can make purchases in inventory as well as make sales from inventory. We need to see that 
the inventory_purchases list is properly increased after every transaction, and the 
inventory_sales are properly decreased. We also want to make sure that our inventory level 
does not drop below 0 (as this is physically not possible). Than we need to make sure we 
can print out our list of customers as well as how many and which items they have 
purchased.




