pragma solidity ^0.4.18; // minimum compiler version

contract HelloCoin {

	//currency name. Please feel free to change it
	string public name = 'HelloCoin'; 
	
	//choose a currency symbol. Please feel free to change it
	string public symbol = 'hc'; 
	
	//a key-value pair to store addresses and their account balances
	mapping (address => uint) balances; 
	
	// declaration of an event. Event will not do anything but add a record to the log
	event Transfer(address _from, address _to, uint256 _value); 

	constructor() public { 
	    // when the contract is created, the constructor will be called automatically
	    balances[msg.sender] = 1000000; 
	    // set the balances of creator account to be 1000000. Please feel free to change it to any number you want.
	}

	function sendCoin(address _receiver, uint _amount) public returns(bool sufficient) {
	    if (balances[msg.sender] < _amount) return false;  
	    // validate transfer
	    balances[msg.sender] -= _amount;
	    balances[_receiver] += _amount;
	    emit Transfer(msg.sender, _receiver, _amount); 
	    // complete coin transfer and call event to record the log
	    return true;
	}

	// getBalance is free because we're not writing on the blockchain
	function getBalance(address _addr) public view returns(uint) { 
	    //balance check
	    return balances[_addr];
	}
}