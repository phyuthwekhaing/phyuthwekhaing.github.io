```python
public class BankAccount{ 
	public string Number { get; } 
	public string Owner { get; set; } 
	public decimal Balance { get; } 
	public void MakeDeposit(decimal amount, DateTime date, string note) { 
	} 
	public void MakeWithdrawal(decimal amount, DateTime date, string 	note) { 
	}
}

```

Add in the constructor for creating a bank account. 


```python
public BankAccount(string name, decimal initialBalance){ 
	this.Owner = name; 
	this.Balance = initialBalance;
}

```
