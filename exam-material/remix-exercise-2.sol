pragma solidity ^0.5.0; // minimum compiler version

contract Coin {
    // The keyword "public" makes those variables easily readable from outside.
    // type (Ethereum) address that everyone can know
    address public minter;

    // mapping is a connection that at every address connects a balance, also
    // public because everyone must know the balance of every address
    mapping (address => uint) public balances;

    // Events allow light clients to react to
    // changes efficiently.
    event Sent(address from, address to, uint amount);

    // This is the constructor whose code is
    // run only when the contract is created.
    constructor() public {
        minter = msg.sender;
    }

    function mint(address receiver, uint amount) public {
        require(msg.sender == minter); // msg.sender is the one interacting with the contract
        require(amount < 1e60);
        balances[receiver] += amount;
    }

    function send2(address receiver1, address receiver2, uint amount) public {
        require(amount <= balances[msg.sender], "Insufficient balance.");
        balances[msg.sender] -= amount;
        balances[receiver1] += amount/2; // important not to do amount*0.5 because might "not be a an integer" whatever that means
        balances[receiver2] += amount/2;
        emit Sent(msg.sender, receiver1, amount/2);

        // if you make an error, you have paid the fees anyways
        // and furthermore the money has disappeared from the sender
    }
}