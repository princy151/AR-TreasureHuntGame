using UnityEngine;

public class HidePanel : MonoBehaviour
{
    public GameObject panelToHide;

    public void Hide()
    {
        panelToHide.SetActive(false);
    }
}